"""Async embed worker pool, exercised against the real (thread-safe) SQLite store."""
import logging
import time

import pytest

from mnemo.adapters.embedding.async_embedding_scheduler import AsyncEmbeddingScheduler
from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.domain.memory import Memory


def _repo(tmp_path):
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.sqlite_vec_repository import SqliteVecMemoryRepository

    return SqliteVecMemoryRepository(path=str(tmp_path / "memory.db"), dim=HashEmbedder().dim)


def _wait(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.01)
    return predicate()


def _pending(repo, content):
    memory = Memory.create(content, project="api")
    repo.add(memory)  # pending, no vector
    return memory


def test_worker_embeds_scheduled_memory(tmp_path):
    repo = _repo(tmp_path)
    scheduler = AsyncEmbeddingScheduler(HashEmbedder(), repo)
    scheduler.start()
    try:
        memory = _pending(repo, "redis cache eviction")
        scheduler.schedule(memory.id)
        assert _wait(lambda: repo.has_vector(memory.id))
        assert repo.pending_count() == 0
    finally:
        scheduler.stop()


def test_recovery_embeds_preexisting_pending_on_start(tmp_path):
    # Pending rows already in the DB (a crash mid-embed, or a migration) get embedded
    # with no explicit schedule() — the workers' first pass drains them. Recovery is free.
    repo = _repo(tmp_path)
    a, b = _pending(repo, "alpha"), _pending(repo, "beta")
    scheduler = AsyncEmbeddingScheduler(HashEmbedder(), repo)
    scheduler.start()
    try:
        assert _wait(lambda: repo.has_vector(a.id) and repo.has_vector(b.id))
    finally:
        scheduler.stop()


def test_backpressure_embeds_inline_at_cap(tmp_path):
    # queue_max=0 → a write always sees the backlog "full" and embeds synchronously,
    # on the caller's thread, without any worker. The queue can never grow unbounded.
    repo = _repo(tmp_path)
    scheduler = AsyncEmbeddingScheduler(HashEmbedder(), repo, queue_max=0)
    memory = _pending(repo, "embed me inline")
    scheduler.schedule(memory.id)  # no workers started — must embed here
    assert repo.has_vector(memory.id) is True


def test_inline_backpressure_logs_a_warning(tmp_path, caplog):
    # The only synchronous (write-blocking) path is the backpressure fallback, so it must
    # be loud: it means a write paid for an encode. queue_max=0 forces it with no workers.
    repo = _repo(tmp_path)
    scheduler = AsyncEmbeddingScheduler(HashEmbedder(), repo, queue_max=0)
    memory = _pending(repo, "inline under backpressure")
    with caplog.at_level(logging.WARNING, logger="mnemo.embed"):
        scheduler.schedule(memory.id)  # embeds on this (caller) thread
    assert any(
        record.levelno == logging.WARNING and "backpressure" in record.getMessage()
        for record in caplog.records
    )


def test_worker_logs_embed_timing_off_the_caller_thread(tmp_path, caplog):
    # The normal path logs the encode at INFO from the worker thread, so the deferred work
    # (and its latency) is observable — and provably NOT on the write/caller thread.
    repo = _repo(tmp_path)
    scheduler = AsyncEmbeddingScheduler(HashEmbedder(), repo)
    scheduler.start()
    try:
        with caplog.at_level(logging.INFO, logger="mnemo.embed"):
            memory = _pending(repo, "log my timing")
            scheduler.schedule(memory.id)
            assert _wait(
                lambda: any("embedded" in r.getMessage() for r in list(caplog.records))
            )
        record = next(r for r in caplog.records if "embedded" in r.getMessage())
        assert record.threadName.startswith("mnemo-embed-")  # a worker, not the caller
    finally:
        scheduler.stop()


def test_drain_waits_for_all_pending(tmp_path):
    repo = _repo(tmp_path)
    scheduler = AsyncEmbeddingScheduler(HashEmbedder(), repo, workers=2)
    scheduler.start()
    try:
        ids = [_pending(repo, f"note {i}").id for i in range(8)]
        for memory_id in ids:
            scheduler.schedule(memory_id)
        scheduler.drain(timeout=3.0)
        assert repo.pending_count() == 0
        assert all(repo.has_vector(memory_id) for memory_id in ids)
    finally:
        scheduler.stop()


def test_transient_failure_is_retried(tmp_path):
    class FlakyEmbedder(HashEmbedder):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def encode(self, text):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient")
            return super().encode(text)

    repo = _repo(tmp_path)
    scheduler = AsyncEmbeddingScheduler(FlakyEmbedder(), repo, max_retries=3)
    scheduler.start()
    try:
        memory = _pending(repo, "retry me")
        scheduler.schedule(memory.id)
        assert _wait(lambda: repo.has_vector(memory.id))
    finally:
        scheduler.stop()


def test_permanent_failure_gives_up_lexical_only(tmp_path):
    class BrokenEmbedder(HashEmbedder):
        def encode(self, text):
            raise RuntimeError("always fails")

    repo = _repo(tmp_path)
    scheduler = AsyncEmbeddingScheduler(BrokenEmbedder(), repo, max_retries=2)
    scheduler.start()
    try:
        memory = _pending(repo, "never embeds")
        scheduler.schedule(memory.id)
        # After max_retries the id is dropped from the work-list (no infinite loop), so
        # drain returns promptly and the memory stays pending (lexical-only).
        scheduler.drain(timeout=3.0)
        assert repo.has_vector(memory.id) is False
        assert repo.pending_count() == 1
    finally:
        scheduler.stop()


def test_failing_encode_retries_exactly_max_retries_then_stops(tmp_path):
    # A fast-failing encode must not spin: it is retried (with backoff) exactly max_retries
    # times, then the id is parked in _failed. Counting the encode calls proves there is no
    # busy-spin past the cap, and that drain waits out the spaced retries before returning.
    class CountingBroken(HashEmbedder):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def encode(self, text):
            self.calls += 1
            raise RuntimeError("always fails")

    embedder = CountingBroken()
    repo = _repo(tmp_path)
    scheduler = AsyncEmbeddingScheduler(embedder, repo, max_retries=3)
    scheduler.start()
    try:
        memory = _pending(repo, "doomed note")
        scheduler.schedule(memory.id)
        scheduler.drain(timeout=8.0)
        assert repo.has_vector(memory.id) is False
        assert embedder.calls == 3  # exactly max_retries — backoff, never a tight loop
    finally:
        scheduler.stop()
