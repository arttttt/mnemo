"""Migration copies every record into the target and is safe to re-run.

The migration is store-agnostic, so the target is parametrized: the in-memory
target runs always (offline), the LanceDB target is marked `heavy`.
"""
import pytest

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.adapters.store.in_memory_repository import InMemoryMemoryRepository
from mnemo.application.use_cases.migrate_memories import MigrateMemories
from mnemo.domain.memory import Memory


def _in_memory_target(tmp_path):
    return InMemoryMemoryRepository(path=str(tmp_path / "target.json"))


def _lancedb_target(tmp_path):
    pytest.importorskip("lancedb")
    from mnemo.adapters.store.lancedb_repository import LanceDbMemoryRepository

    return LanceDbMemoryRepository(uri=str(tmp_path / "memory"))


@pytest.fixture(
    params=[
        pytest.param(_in_memory_target, id="to_in_memory"),
        pytest.param(_lancedb_target, id="to_lancedb", marks=pytest.mark.heavy),
    ]
)
def target(request, tmp_path):
    return request.param(tmp_path)


def _seed_source(tmp_path, embedder, contents):
    source = InMemoryMemoryRepository(path=str(tmp_path / "source.json"))
    seeded = []
    for content in contents:
        memory = Memory.create(content, project="api")
        source.add(memory, embedder.encode(memory.content))
        seeded.append(memory)
    return source, seeded


def test_migrates_all_records(tmp_path, target):
    embedder = HashEmbedder()
    source, seeded = _seed_source(tmp_path, embedder, ["redis cache", "postgres pool", "jwt"])

    result = MigrateMemories(source, target, embedder).execute()

    assert (result.source_total, result.added, result.skipped) == (3, 3, 0)
    assert {m.id for m in target.list_all()} == {m.id for m in seeded}


def test_migration_is_idempotent(tmp_path, target):
    embedder = HashEmbedder()
    source, _ = _seed_source(tmp_path, embedder, ["one", "two"])
    migrate = MigrateMemories(source, target, embedder)

    migrate.execute()
    second = migrate.execute()

    assert (second.added, second.skipped) == (0, 2)
    assert len(target.list_all()) == 2
