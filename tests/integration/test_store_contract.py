"""One contract, run against every MemoryRepositoryPort backend.

The in-memory and SQLite (`sqlite-vec` + FTS5) backends both run always — they
are offline and light (`sqlite-vec` is a small extension, skipped gracefully if
absent).
"""
import pytest

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.application.retrieval import Retrieval
from mnemo.application.search_criteria import SearchCriteria
from mnemo.domain.memory import Memory

_ALL = SearchCriteria(scope="all")


def _hits(repo, embedder, text, criteria, limit=5):
    """Run a semantic retrieval the way the use case does (text + its embedding)."""
    return repo.retrieve(
        Retrieval(criteria=criteria, limit=limit, text=text, vector=embedder.encode(text))
    )


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


def test_add_and_find_active_by_hash(open_repo, embedder):
    repo = open_repo()
    memory = _store(repo, embedder, "durable note", type="decision", project="api")

    found = repo.find_active_by_hash(memory.hash, "api")
    assert found is not None and found.id == memory.id
    assert found.type == memory.type and found.content == "durable note"
    assert repo.find_active_by_hash("does-not-exist", "api") is None


def test_find_active_by_hash_is_scoped_to_project(open_repo, embedder):
    """Identical content is a DISTINCT memory in another project — the hash key is
    global but content is unique only within a scope, so the lookup is project-scoped."""
    repo = open_repo()
    here = _store(repo, embedder, "shared fact", project="api")
    there = _store(repo, embedder, "shared fact", project="other")  # same hash, kept

    assert here.hash == there.hash  # the content hash itself is project-independent
    assert repo.find_active_by_hash(here.hash, "api").id == here.id
    assert repo.find_active_by_hash(here.hash, "other").id == there.id
    assert repo.find_active_by_hash(here.hash, "absent") is None


def test_find_active_by_hash_ignores_superseded(open_repo, embedder):
    """Superseded rows are not retrievable, so they must not match the dedup lookup —
    otherwise re-storing the content would return a dead id and never re-create it."""
    repo = open_repo()
    memory = _store(repo, embedder, "evolving note", project="api")
    repo.mark_superseded(memory.id)

    assert repo.find_active_by_hash(memory.hash, "api") is None


def test_persists_across_reopen(open_repo, embedder):
    memory = _store(
        open_repo(), embedder, "remembered after reopen", project="api", session_id="run-1"
    )

    reopened = open_repo()
    stored = reopened.list_all()
    assert [m.id for m in stored] == [memory.id]
    assert stored[0].session_id == "run-1"  # session_id round-trips through the store
    assert reopened.find_active_by_hash(memory.hash, "api") is not None


def test_search_ranks_by_similarity(open_repo, embedder):
    repo = open_repo()
    for content in ["redis caching layer", "postgres migration plan", "redis cache eviction"]:
        _store(repo, embedder, content, project="api")

    hits = _hits(repo, embedder, "redis cache", _ALL, limit=3)
    assert hits[0].score >= hits[-1].score
    assert "redis" in hits[0].memory.content


def test_search_scopes_to_project(open_repo, embedder):
    repo = open_repo()
    keep = _store(repo, embedder, "redis cache eviction", project="api")
    _store(repo, embedder, "redis cache eviction notes", project="other")

    criteria = SearchCriteria(scope="project", project="api")
    hits = _hits(repo, embedder, "redis cache", criteria, limit=5)
    assert [hit.memory.id for hit in hits] == [keep.id]


def test_search_filters_by_tags_and_files(open_repo, embedder):
    repo = open_repo()
    tagged = _store(
        repo, embedder, "jwt rotation policy", project="api",
        tags=["auth", "jwt"], related_files=["src/auth/jwt.ts"],
    )
    _store(repo, embedder, "jwt rotation note", project="api", tags=["auth"])

    by_tags = _hits(
        repo, embedder, "jwt", SearchCriteria(scope="all", tags=("auth", "jwt")), limit=5
    )
    assert [hit.memory.id for hit in by_tags] == [tagged.id]

    by_file = _hits(
        repo,
        embedder,
        "jwt",
        SearchCriteria(scope="all", related_files=("src/auth/jwt.ts",)),
        limit=5,
    )
    assert [hit.memory.id for hit in by_file] == [tagged.id]


def test_search_recency_excludes_old(open_repo, embedder):
    repo = open_repo()
    _store(repo, embedder, "fresh note", project="api")

    future_cutoff = SearchCriteria(scope="all", created_after="2999-01-01T00:00:00+00:00")
    assert _hits(repo, embedder, "fresh note", future_cutoff, limit=5) == []


def test_created_after_filters_by_utc_instant(open_repo, embedder):
    """A non-UTC-offset created_after is normalized to UTC, so it filters by instant
    through both the in-process matcher and the SQL `>=` string comparison."""
    repo = open_repo()
    early = Memory.create("early note", project="api")
    early.created_at = "2026-06-19T08:00:00+00:00"
    repo.add(early, embedder.encode(early.content))
    late = Memory.create("late note", project="api")
    late.created_at = "2026-06-19T11:00:00+00:00"
    repo.add(late, embedder.encode(late.content))

    # bound 12:00+03:00 == 09:00 UTC → only the 11:00 memory is at/after it
    criteria = SearchCriteria(scope="all", created_after="2026-06-19T12:00:00+03:00")
    hits = repo.retrieve(Retrieval(criteria=criteria, limit=10))  # filter-only browse
    assert {hit.memory.id for hit in hits} == {late.id}


def test_browse_lists_by_recency_with_no_score(open_repo, embedder):
    """A retrieval with no text and no vector is a filter-only browse: newest
    first, no relevance ranking (score 0.0), no embedding needed."""
    repo = open_repo()
    one = _store(repo, embedder, "first note", project="api")
    two = _store(repo, embedder, "second note", project="api")
    three = _store(repo, embedder, "third note", project="api")

    hits = repo.retrieve(Retrieval(criteria=_ALL, limit=10))
    created = [hit.memory.created_at for hit in hits]
    assert created == sorted(created, reverse=True)  # newest first
    assert {hit.memory.id for hit in hits} == {one.id, two.id, three.id}
    assert all(hit.score == 0.0 for hit in hits)  # order conveys recency, not a score


def test_browse_respects_scope_and_includes_pending(open_repo, embedder):
    repo = open_repo()
    mine = _store(repo, embedder, "my project note", project="api")
    _store(repo, embedder, "other project note", project="other")
    pending = Memory.create("pending browse note", project="api")
    repo.add(pending)  # no vector — browse still surfaces it (no vector needed)

    criteria = SearchCriteria(scope="project", project="api")
    hits = repo.retrieve(Retrieval(criteria=criteria, limit=10))
    assert {hit.memory.id for hit in hits} == {mine.id, pending.id}


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
    assert repo.find_active_by_hash(pending.hash, "api") is not None
    assert pending.id in {m.id for m in repo.list_all()}

    repo.set_vector(pending.id, embedder.encode(pending.content))

    assert repo.has_vector(pending.id) is True
    assert repo.pending_count() == 0
    assert repo.next_unembedded(10) == []
    hits = _hits(repo, embedder, "redis", _ALL, limit=5)
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

    hits = _hits(repo, embedder, "handleAuthCallback", _ALL, limit=5)
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
    hits = _hits(repo, embedder, "handleAuthCallback", _ALL, limit=3)
    assert hits and hits[0].memory.id == target.id


def test_find_active_by_topic_key(open_repo, embedder):
    repo = open_repo()
    memory = _store(repo, embedder, "auth model", project="api", topic_key="auth/model")

    found = repo.find_active_by_topic_key("auth/model", "api")
    assert found is not None and found.id == memory.id
    assert repo.find_active_by_topic_key("auth/model", "other") is None
    assert repo.find_active_by_topic_key("absent/key", "api") is None


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
