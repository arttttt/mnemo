"""One contract, run against every MemoryRepositoryPort backend.

The in-memory backend runs always (offline). The LanceDB backend is marked
`heavy` so it is skipped by the default offline run and exercised only with
`-m heavy` (it needs the optional `lancedb` dependency).
"""
import pytest

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.application.search_criteria import SearchCriteria
from mnemo.domain.memory import Memory

_ALL = SearchCriteria(scope="all")


def _in_memory(tmp_path):
    from mnemo.adapters.store.in_memory_repository import InMemoryMemoryRepository

    return InMemoryMemoryRepository(path=str(tmp_path / "memory.json"))


def _lancedb(tmp_path):
    pytest.importorskip("lancedb")
    from mnemo.adapters.store.lancedb_repository import LanceDbMemoryRepository

    return LanceDbMemoryRepository(uri=str(tmp_path / "lancedb"))


@pytest.fixture(
    params=[
        pytest.param(_in_memory, id="in_memory"),
        pytest.param(_lancedb, id="lancedb", marks=pytest.mark.heavy),
    ]
)
def open_repo(request, tmp_path):
    """Return a zero-arg factory that (re)opens a repo at one fixed location."""
    return lambda: request.param(tmp_path)


@pytest.fixture
def embedder():
    return HashEmbedder()


def _store(repo, embedder, content, **kwargs):
    memory = Memory.create(content, **kwargs)
    repo.add(memory, embedder.encode(memory.content))
    return memory


def test_add_and_find_by_hash(open_repo, embedder):
    repo = open_repo()
    memory = _store(repo, embedder, "durable note", type="decision", project="api")

    found = repo.find_by_hash(memory.hash)
    assert found is not None and found.id == memory.id
    assert found.type == memory.type and found.content == "durable note"
    assert repo.find_by_hash("does-not-exist") is None


def test_persists_across_reopen(open_repo, embedder):
    memory = _store(
        open_repo(), embedder, "remembered after reopen", project="api", session_id="run-1"
    )

    reopened = open_repo()
    stored = reopened.list_all()
    assert [m.id for m in stored] == [memory.id]
    assert stored[0].session_id == "run-1"  # session_id round-trips through the store
    assert reopened.find_by_hash(memory.hash) is not None


def test_search_ranks_by_similarity(open_repo, embedder):
    repo = open_repo()
    for content in ["redis caching layer", "postgres migration plan", "redis cache eviction"]:
        _store(repo, embedder, content, project="api")

    hits = repo.search("redis cache", embedder.encode("redis cache"), _ALL, limit=3)
    assert hits[0].score >= hits[-1].score
    assert "redis" in hits[0].memory.content


def test_search_scopes_to_project(open_repo, embedder):
    repo = open_repo()
    keep = _store(repo, embedder, "redis cache eviction", project="api")
    _store(repo, embedder, "redis cache eviction notes", project="other")

    criteria = SearchCriteria(scope="project", project="api")
    hits = repo.search("redis cache", embedder.encode("redis cache"), criteria, limit=5)
    assert [hit.memory.id for hit in hits] == [keep.id]


def test_search_filters_by_tags_and_files(open_repo, embedder):
    repo = open_repo()
    tagged = _store(
        repo, embedder, "jwt rotation policy", project="api",
        tags=["auth", "jwt"], related_files=["src/auth/jwt.ts"],
    )
    _store(repo, embedder, "jwt rotation note", project="api", tags=["auth"])

    by_tags = repo.search(
        "jwt", embedder.encode("jwt"), SearchCriteria(scope="all", tags=("auth", "jwt")), limit=5
    )
    assert [hit.memory.id for hit in by_tags] == [tagged.id]

    by_file = repo.search(
        "jwt",
        embedder.encode("jwt"),
        SearchCriteria(scope="all", related_files=("src/auth/jwt.ts",)),
        limit=5,
    )
    assert [hit.memory.id for hit in by_file] == [tagged.id]


def test_search_recency_excludes_old(open_repo, embedder):
    repo = open_repo()
    _store(repo, embedder, "fresh note", project="api")

    future_cutoff = SearchCriteria(scope="all", created_after="2999-01-01T00:00:00+00:00")
    assert repo.search("fresh note", embedder.encode("fresh note"), future_cutoff, limit=5) == []


@pytest.mark.heavy
def test_lancedb_hybrid_finds_exact_token(tmp_path):
    pytest.importorskip("lancedb")
    from mnemo.adapters.store.lancedb_repository import LanceDbMemoryRepository

    embedder = HashEmbedder()
    repo = LanceDbMemoryRepository(uri=str(tmp_path / "memory"))
    target = _store(repo, embedder, "the fix lives in handleAuthCallback", project="api")
    _store(repo, embedder, "unrelated postgres migration notes", project="api")

    # The full-text index is created with the table, so an exact token ranks
    # first via the lexical half of the hybrid.
    hits = repo.search(
        "handleAuthCallback", embedder.encode("handleAuthCallback"), _ALL, limit=3
    )
    assert hits and hits[0].memory.id == target.id


def test_find_active_by_topic_key(open_repo, embedder):
    repo = open_repo()
    memory = _store(repo, embedder, "auth model", project="api", topic_key="auth/model")

    found = repo.find_active_by_topic_key("auth/model", "api")
    assert found is not None and found.id == memory.id
    assert repo.find_active_by_topic_key("auth/model", "other") is None
    assert repo.find_active_by_topic_key("absent/key", "api") is None


def test_register_duplicate_increments_count(open_repo, embedder):
    repo = open_repo()
    memory = _store(repo, embedder, "seen twice", project="api")

    repo.register_duplicate(memory.id)
    assert repo.find_by_hash(memory.hash).duplicate_count == 1


def test_mark_superseded_sets_status(open_repo, embedder):
    repo = open_repo()
    memory = _store(repo, embedder, "old version", project="api")

    repo.mark_superseded(memory.id)
    status_by_id = {m.id: m.status for m in repo.list_all()}
    assert status_by_id[memory.id] == "superseded"


def test_delete_clear_purge(open_repo, embedder):
    repo = open_repo()
    one = _store(repo, embedder, "one", project="api")
    _store(repo, embedder, "two", project="api")
    _store(repo, embedder, "three", project="other")

    assert repo.delete([one.id]) == 1
    assert repo.delete_by_project("api") == 1
    assert {m.project for m in repo.list_all()} == {"other"}
    assert repo.delete_all() == 1
    assert repo.list_all() == []
