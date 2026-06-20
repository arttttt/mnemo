import pytest

pytest.importorskip("sqlite_vec")

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.adapters.embedding.sync_embedding_scheduler import SyncEmbeddingScheduler
from mnemo.domain.memory import Memory
from tests.support.sqlite_store import open_store


def _setup(tmp_path):
    embedder = HashEmbedder()
    repo, _ = open_store(tmp_path, embedder.dim, projects=("api",))
    return repo, embedder, SyncEmbeddingScheduler(embedder, repo)


def test_schedule_embeds_inline(tmp_path):
    repo, _, scheduler = _setup(tmp_path)
    memory = Memory.create("redis cache eviction", project="api")
    repo.add(memory)  # pending, no vector
    assert repo.has_vector(memory.id) is False

    scheduler.schedule(memory.id)

    assert repo.has_vector(memory.id) is True
    assert repo.pending_count() == 0


def test_schedule_on_missing_id_is_noop(tmp_path):
    _, _, scheduler = _setup(tmp_path)
    scheduler.schedule("does-not-exist")  # deleted before embedding — no raise
