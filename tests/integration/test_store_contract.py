"""One contract, run against every MemoryRepositoryPort backend.

The in-memory and SQLite (`sqlite-vec` + FTS5) backends both run always — they
are offline and light (`sqlite-vec` is a small extension, skipped gracefully if
absent).
"""
import pytest

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.application.search_criteria import SearchCriteria
from mnemo.domain.memory import Memory

_ALL = SearchCriteria(scope="all")


def _in_memory(tmp_path):
    from mnemo.adapters.store.in_memory_repository import InMemoryMemoryRepository

    return InMemoryMemoryRepository(path=str(tmp_path / "memory.json"))


def _sqlite(tmp_path):
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.sqlite_vec_repository import SqliteVecMemoryRepository

    # dim up front so a pending (vector-less) write can create the schema.
    return SqliteVecMemoryRepository(path=str(tmp_path / "memory.db"), dim=HashEmbedder().dim)


@pytest.fixture(
    params=[
        pytest.param(_in_memory, id="in_memory"),
        pytest.param(_sqlite, id="sqlite"),
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


def test_pending_vector_lifecycle(open_repo, embedder):
    """Deferred embedding: a memory can be stored without a vector (pending), then
    have it attached later — it only enters dense search once set_vector lands."""
    repo = open_repo()
    pending = Memory.create("a pending memory about redis", type="decision", project="api")
    repo.add(pending)  # no vector → pending

    assert repo.pending_count() == 1
    assert repo.next_unembedded(10) == [pending.id]
    assert repo.has_vector(pending.id) is False
    assert repo.content_for(pending.id) == "a pending memory about redis"
    # stored and addressable even without a vector
    assert repo.find_by_hash(pending.hash) is not None
    assert pending.id in {m.id for m in repo.list_all()}

    repo.set_vector(pending.id, embedder.encode(pending.content))

    assert repo.has_vector(pending.id) is True
    assert repo.pending_count() == 0
    assert repo.next_unembedded(10) == []
    hits = repo.search("redis", embedder.encode("redis"), _ALL, limit=5)
    assert any(hit.memory.id == pending.id for hit in hits)  # now in dense search


def test_missing_id_methods_are_safe(open_repo, embedder):
    repo = open_repo()
    assert repo.has_vector("nope") is False
    assert repo.content_for("nope") is None
    repo.set_vector("nope", embedder.encode("x"))  # no-op, no raise


def test_sqlite_pending_is_lexically_searchable(tmp_path):
    """A pending memory (no vector) must still be findable via the FTS5 lexical leg."""
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.sqlite_vec_repository import SqliteVecMemoryRepository

    embedder = HashEmbedder()
    repo = SqliteVecMemoryRepository(path=str(tmp_path / "memory.db"), dim=embedder.dim)
    pending = Memory.create("handleAuthCallback pending fix", project="api")
    repo.add(pending)  # no vector

    hits = repo.search(
        "handleAuthCallback", embedder.encode("handleAuthCallback"), _ALL, limit=5
    )
    assert any(hit.memory.id == pending.id for hit in hits)


def test_sqlite_pending_first_write_without_dim_errors(tmp_path):
    """Without a known dimension a vector-less first write cannot create the schema."""
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.sqlite_vec_repository import SqliteVecMemoryRepository

    repo = SqliteVecMemoryRepository(path=str(tmp_path / "memory.db"))  # no dim
    with pytest.raises(ValueError):
        repo.add(Memory.create("pending with no dim", project="api"))


def test_sqlite_hybrid_finds_exact_token(tmp_path):
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.sqlite_vec_repository import SqliteVecMemoryRepository

    embedder = HashEmbedder()
    repo = SqliteVecMemoryRepository(path=str(tmp_path / "memory.db"))
    target = _store(repo, embedder, "the fix lives in handleAuthCallback", project="api")
    _store(repo, embedder, "unrelated postgres migration notes", project="api")

    # FTS5 is created with the schema, so an exact token ranks first via the
    # lexical (BM25) half of the hybrid, even though it is a rare term.
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
