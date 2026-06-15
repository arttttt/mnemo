from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.adapters.embedding.sync_embedding_scheduler import SyncEmbeddingScheduler
from mnemo.adapters.store.in_memory_repository import InMemoryMemoryRepository
from mnemo.domain.memory import Memory


def _setup():
    repo = InMemoryMemoryRepository()
    embedder = HashEmbedder()
    return repo, embedder, SyncEmbeddingScheduler(embedder, repo)


def test_schedule_embeds_inline():
    repo, _, scheduler = _setup()
    memory = Memory.create("redis cache eviction", project="api")
    repo.add(memory)  # pending, no vector
    assert repo.has_vector(memory.id) is False

    scheduler.schedule(memory.id)

    assert repo.has_vector(memory.id) is True
    assert repo.pending_count() == 0


def test_schedule_on_missing_id_is_noop():
    _, _, scheduler = _setup()
    scheduler.schedule("does-not-exist")  # deleted before embedding — no raise


def test_window_methods_delegate_to_embedder():
    _, embedder, scheduler = _setup()
    assert scheduler.max_input == embedder.max_input
    assert scheduler.count_tokens("one two three") == embedder.count_tokens("one two three")
