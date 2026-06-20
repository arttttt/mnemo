"""Deferred embedding: a pool of worker threads computes vectors off the hot path.

The **database is the durable queue** — a memory with no vector is a pending job, so
there is no separate queue store and recovery is free (a worker's first pass drains
whatever is pending). `schedule(id)` only wakes the workers; the row is already
pending in the DB. The design (event-driven, backpressure, retries, drain) is in
docs/03-architecture.md.

Threads suit this: onnxruntime releases the GIL during the forward pass, so encoding
does not block the service handling MCP requests.
"""
from __future__ import annotations

import logging
import threading
import time

from mnemo.application.ports.embedder import TextEmbedder
from mnemo.application.ports.embedding_queue import EmbeddingQueue

_log = logging.getLogger("mnemo.embed")

# A failed encode is retried with exponential backoff so a fast-failing embedder can't
# spin a worker (or drain) through its retries back-to-back, burning CPU.
_RETRY_BACKOFF_BASE = 0.5
_RETRY_BACKOFF_MAX = 30.0


class AsyncEmbeddingScheduler:
    def __init__(
        self,
        embedder: TextEmbedder,
        repository: EmbeddingQueue,
        *,
        workers: int = 1,
        queue_max: int = 256,
        max_retries: int = 3,
    ) -> None:
        self._embedder = embedder
        self._repository = repository
        self._queue_max = queue_max
        self._max_retries = max_retries
        self._cond = threading.Condition()
        self._stopping = False
        self._in_flight = 0
        self._claimed: set[str] = set()      # ids a worker (or an inline encode) is embedding
        self._retries: dict[str, int] = {}   # id -> failed attempts so far
        self._failed: set[str] = set()        # gave up after max_retries (stays lexical-only)
        self._retry_after: dict[str, float] = {}  # id -> monotonic time before which it can't retry
        self._threads = [
            threading.Thread(target=self._run, name=f"mnemo-embed-{i}", daemon=True)
            for i in range(max(1, workers))
        ]

    # --- lifecycle ---
    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    def drain(self, timeout: float) -> None:
        """Block until all pending work is embedded (or timeout) — used before idle-exit.

        Assumes writes have stopped (the idle monitor only drains once no connector is
        alive), so the work set only shrinks. The pending DB scan runs OUTSIDE the lock so
        it never blocks the workers; only the in-memory counters are read under it.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._cond:
                busy = self._in_flight > 0 or self._has_pending_retry()
            # `_first_claimable()` does the DB scan; membership/get on the shared sets are
            # GIL-atomic, so calling it without the lock is safe (and never blocks a worker).
            if not busy and self._first_claimable() is None:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            with self._cond:
                self._cond.wait(min(0.1, remaining))  # bounded re-check of the DB predicate

    def stop(self) -> None:
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        for thread in self._threads:
            thread.join(timeout=1.0)

    # --- scheduling ---
    def schedule(self, memory_id: str) -> None:
        # Backpressure: keep the pending backlog bounded. At the cap, embed inline so a
        # write can never grow an unbounded queue — worst case it is as slow as one encode.
        pending = self._repository.pending_count()
        if pending >= self._queue_max:
            _log.warning(
                "backpressure: %d pending at cap %d — embedding %s inline (this blocks the write)",
                pending, self._queue_max, memory_id,
            )
            self._embed_inline(memory_id)
            return
        _log.debug("queued %s (pending=%d); waking a worker", memory_id, pending)
        with self._cond:
            self._cond.notify()

    def _embed_inline(self, memory_id: str) -> None:
        # Backpressure fallback on the caller's thread. Account for it in `_in_flight` and
        # `_claimed` like the worker path, so a concurrent `drain()` waits for it and no
        # worker double-encodes the same id.
        with self._cond:
            if memory_id in self._claimed:
                return  # a worker already has it
            self._in_flight += 1
            self._claimed.add(memory_id)
        try:
            self._embed_one(memory_id)
        finally:
            with self._cond:
                self._in_flight -= 1
                self._claimed.discard(memory_id)
                self._cond.notify_all()

    # --- worker ---
    def _run(self) -> None:
        while True:
            with self._cond:
                while True:
                    if self._stopping and self._in_flight == 0 and self._first_claimable() is None:
                        return
                    memory_id = self._claim_next()
                    if memory_id is not None:
                        self._in_flight += 1
                        break
                    if self._stopping:
                        return
                    self._cond.wait()  # woken by schedule(), a completion, or a retry timer
            try:
                self._embed_one(memory_id)
            finally:
                with self._cond:
                    self._in_flight -= 1
                    self._claimed.discard(memory_id)
                    self._cond.notify_all()  # wake drain() and idle workers

    def _first_claimable(self) -> str | None:
        """The next pending id no worker holds, not permanently failed, and past its retry
        backoff. Reads are GIL-atomic, so this is safe with OR without `self._cond`."""
        now = time.monotonic()
        for memory_id in self._repository.next_unembedded(self._queue_max):
            if memory_id in self._claimed or memory_id in self._failed:
                continue
            retry_at = self._retry_after.get(memory_id)
            if retry_at is not None and retry_at > now:
                continue  # backing off — not eligible to retry yet
            return memory_id
        return None

    def _claim_next(self) -> str | None:
        memory_id = self._first_claimable()
        if memory_id is not None:
            self._claimed.add(memory_id)
        return memory_id

    def _has_pending_retry(self) -> bool:
        """Caller holds `self._cond`. True while some id is waiting out its retry backoff
        (so drain keeps waiting for the retry instead of returning early)."""
        return any(memory_id not in self._failed for memory_id in self._retry_after)

    def _embed_one(self, memory_id: str) -> None:
        if self._repository.has_vector(memory_id):
            return  # already done (retry-after-success, race, or recovery overlap)
        content = self._repository.content_for(memory_id)
        if content is None:
            return  # deleted before embedding — nothing to do
        start = time.monotonic()
        try:
            vector = self._embedder.encode(content)
        except Exception:  # noqa: BLE001 — any encode failure is retried, never fatal
            self._record_failure(memory_id)
            return
        self._repository.set_vector(memory_id, vector)
        with self._cond:  # bookkeeping mutation stays under the lock
            self._retries.pop(memory_id, None)
            self._retry_after.pop(memory_id, None)
        _log.info(
            "embedded %s in %d ms (%d chars)",
            memory_id, int((time.monotonic() - start) * 1000), len(content),
        )

    def _record_failure(self, memory_id: str) -> None:
        with self._cond:
            attempts = self._retries.get(memory_id, 0) + 1
            self._retries[memory_id] = attempts
            if attempts >= self._max_retries:
                self._failed.add(memory_id)
                self._retry_after.pop(memory_id, None)
                self._cond.notify_all()  # let drain observe the give-up
                _log.warning(
                    "embedding permanently failed for %s after %d attempts (stays lexical-only)",
                    memory_id, attempts,
                )
                return
            backoff = min(_RETRY_BACKOFF_BASE * 2 ** (attempts - 1), _RETRY_BACKOFF_MAX)
            self._retry_after[memory_id] = time.monotonic() + backoff
        # Wake a worker once the backoff elapses (one-shot timer → retries are spaced, and
        # the worker loop stays purely notify-driven).
        timer = threading.Timer(backoff, self._wake)
        timer.daemon = True
        timer.start()

    def _wake(self) -> None:
        with self._cond:
            self._cond.notify_all()
