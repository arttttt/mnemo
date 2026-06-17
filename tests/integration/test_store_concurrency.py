"""The SQLite store is safe under concurrent readers and writers.

Validates the single-writer + per-thread-reader model at the adapter level (a
precursor to the full multi-agent service stress).
"""
import concurrent.futures as cf

import pytest

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.application.retrieval import Retrieval
from mnemo.application.search_criteria import SearchCriteria
from mnemo.domain.memory import Memory

_ALL = SearchCriteria(scope="all")


def test_concurrent_writes_and_reads_lose_nothing(tmp_path):
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.sqlite_vec_repository import SqliteVecMemoryRepository

    repo = SqliteVecMemoryRepository(path=str(tmp_path / "memory.db"))
    embedder = HashEmbedder()
    writers, per_writer = 8, 10

    def write(w):
        for i in range(per_writer):
            memory = Memory.create(f"note {w}-{i} redis cache", project="api")
            repo.add(memory, embedder.encode(memory.content))

    def read(_):
        # Runs alongside the writers; must not error (and never sees a torn DB).
        return repo.retrieve(
            Retrieval(criteria=_ALL, limit=5, text="redis cache",
                      vector=embedder.encode("redis cache"))
        )

    with cf.ThreadPoolExecutor(max_workers=writers * 2) as pool:
        tasks = [pool.submit(write, w) for w in range(writers)]
        tasks += [pool.submit(read, r) for r in range(writers)]
        for task in cf.as_completed(tasks):
            task.result()  # re-raise any error from a worker

    assert len(repo.list_all()) == writers * per_writer  # zero lost writes
