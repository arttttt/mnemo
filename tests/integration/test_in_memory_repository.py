from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.adapters.store.in_memory_repository import InMemoryMemoryRepository
from mnemo.domain.memory import Memory


def test_persists_and_reloads_from_disk(tmp_path):
    path = tmp_path / "memory.json"
    embedder = HashEmbedder()

    repo = InMemoryMemoryRepository(path=str(path))
    memory = Memory.create("durable note", type="decision", project="api")
    repo.add(memory, embedder.encode(memory.content))

    reloaded = InMemoryMemoryRepository(path=str(path))
    stored = reloaded.list_all()
    assert len(stored) == 1
    assert stored[0].id == memory.id
    assert stored[0].type == memory.type
    assert stored[0].content == "durable note"
    assert reloaded.find_by_hash(memory.hash) is not None


def test_search_ranks_by_cosine():
    embedder = HashEmbedder()
    repo = InMemoryMemoryRepository()
    for content in ["redis caching layer", "postgres migration plan", "redis cache eviction"]:
        memory = Memory.create(content, project="api")
        repo.add(memory, embedder.encode(memory.content))

    hits = repo.search(embedder.encode("redis cache"), limit=3)
    assert hits[0].score >= hits[-1].score
    assert "redis" in hits[0].memory.content


def test_find_active_by_topic_key(tmp_path):
    embedder = HashEmbedder()
    repo = InMemoryMemoryRepository()
    memory = Memory.create("auth model", project="api", topic_key="auth/model")
    repo.add(memory, embedder.encode(memory.content))

    found = repo.find_active_by_topic_key("auth/model", "api")
    assert found is not None and found.id == memory.id
    assert repo.find_active_by_topic_key("auth/model", "other") is None
