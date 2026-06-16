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

from mnemo.application.ports.embedder import EmbedderPort
from mnemo.application.ports.memory_repository import MemoryRepositoryPort

_log = logging.getLogger("mnemo.embed")


class AsyncEmbeddingScheduler:
    def __init__(
        self,
        embedder: EmbedderPort,
        repository: MemoryRepositoryPort,
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
        self._claimed: set[str] = set()      # ids a worker is currently embedding
        self._retries: dict[str, int] = {}   # id -> failed attempts so far
        self._failed: set[str] = set()        # gave up after max_retries (stays lexical-only)
        self._threads = [
            threading.Thread(target=self._run, name=f"mnemo-embed-{i}", daemon=True)
            for i in range(max(1, workers))
        ]

    # --- window-check delegation (the use case rejects oversize before insert) ---
    @property
    def max_input(self) -> int:
        return self._embedder.max_input

    def count_tokens(self, text: str) -> int:
        return self._embedder.count_tokens(text)

    # --- lifecycle ---
    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    def drain(self, timeout: float) -> None:
        """Block until all pending work is embedded (or timeout) — used before idle-exit."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._in_flight > 0 or self._first_claimable() is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._cond.wait(remaining)

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
        if self._repository.pending_count() >= self._queue_max:
            self._embed_one(memory_id)
            return
        with self._cond:
            self._cond.notify()

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
                    self._cond.wait()
            try:
                self._embed_one(memory_id)
            finally:
                with self._cond:
                    self._in_flight -= 1
                    self._claimed.discard(memory_id)
                    self._cond.notify_all()  # wake drain() and idle workers

    def _first_claimable(self) -> str | None:
        """The next pending id no worker holds and that hasn't permanently failed.
        Caller holds `self._cond`."""
        for memory_id in self._repository.next_unembedded(self._queue_max):
            if memory_id not in self._claimed and memory_id not in self._failed:
                return memory_id
        return None

    def _claim_next(self) -> str | None:
        memory_id = self._first_claimable()
        if memory_id is not None:
            self._claimed.add(memory_id)
        return memory_id

    def _embed_one(self, memory_id: str) -> None:
        if self._repository.has_vector(memory_id):
            return  # already done (retry-after-success, race, or recovery overlap)
        content = self._repository.content_for(memory_id)
        if content is None:
            return  # deleted before embedding — nothing to do
        try:
            vector = self._embedder.encode(content)
        except Exception:  # noqa: BLE001 — any encode failure is retried, never fatal
            self._record_failure(memory_id)
            return
        self._repository.set_vector(memory_id, vector)
        self._retries.pop(memory_id, None)

    def _record_failure(self, memory_id: str) -> None:
        attempts = self._retries.get(memory_id, 0) + 1
        self._retries[memory_id] = attempts
        if attempts >= self._max_retries:
            self._failed.add(memory_id)
            _log.warning(
                "embedding permanently failed for %s after %d attempts (stays lexical-only)",
                memory_id, attempts,
            )
        else:
            with self._cond:  # leave it pending; nudge a worker to retry
                self._cond.notify()
