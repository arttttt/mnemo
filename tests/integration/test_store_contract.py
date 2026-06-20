"""The MemoryRepository contract, exercised against the SQLite (`sqlite-vec` +
FTS5) backend — the sole store. Offline and light; skipped where `sqlite-vec` is absent.
"""
import pytest

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.application.retrieval import Retrieval
from mnemo.application.search_criteria import SearchCriteria
from mnemo.domain.memory import Memory
from tests.support.sqlite_store import open_store

_ALL = SearchCriteria(scope="all")


def _hits(repo, embedder, text, criteria, limit=5):
    """Run a semantic retrieval the way the use case does (text + its embedding)."""
    return repo.retrieve(
        Retrieval(criteria=criteria, limit=limit, text=text, vector=embedder.encode(text))
    )


def _sqlite(tmp_path):
    # Register the projects these tests write to so inserts satisfy the FK
    # (memories.project -> projects.slug). Idempotent, so reopen is fine.
    repo, _ = open_store(tmp_path, HashEmbedder().dim, projects=("api", "other"))
    return repo


@pytest.fixture
def open_repo(tmp_path):
    """Return a zero-arg factory that (re)opens a repo at one fixed location."""
    return lambda: _sqlite(tmp_path)


@pytest.fixture
def embedder():
    return HashEmbedder()


def _store(repo, embedder, content, **kwargs):
    memory = Memory.create(content, **kwargs)
    repo.add(memory, embedder.encode(memory.content))
    return memory


def _supersede(repo, embedder, prior):
    """Drive `prior` into the superseded state via the production supersede path
    (a test helper — the store exposes no test-only 'mark superseded' shortcut)."""
    successor = Memory.create(
        f"successor of {prior.content}", project=prior.project, topic_key="evolved"
    )
    successor.supersedes = prior.id
    repo.supersede(successor, embedder.encode(successor.content))
    return successor


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
    _supersede(repo, embedder, memory)

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
    embedder = HashEmbedder()
    repo, _ = open_store(tmp_path, embedder.dim, projects=("api",))
    pending = Memory.create("handleAuthCallback pending fix", project="api")
    repo.add(pending)  # no vector

    hits = _hits(repo, embedder, "handleAuthCallback", _ALL, limit=5)
    assert any(hit.memory.id == pending.id for hit in hits)


def test_sqlite_deleting_a_project_cascades_its_memories(tmp_path):
    """The FK cascade is the whole delete_project mechanism: removing the project row
    atomically removes its memories (memories.project FK)."""
    embedder = HashEmbedder()
    repo, registry = open_store(tmp_path, embedder.dim, projects=("api", "other"))
    first = _store(repo, embedder, "auth model v1", project="api", topic_key="auth/model")
    _supersede(repo, embedder, first)  # a second memory in `api` (a supersede chain)
    kept = _store(repo, embedder, "kept elsewhere", project="other")

    registry.delete("api")  # DELETE FROM projects -> cascades the api memories

    assert {m.id for m in repo.list_all()} == {kept.id}  # only the other project survives


def test_sqlite_memory_for_an_unregistered_project_is_rejected(tmp_path):
    """The FK also makes the gate a DB invariant: no memory can reference a project
    that was never registered."""
    import sqlite3

    embedder = HashEmbedder()
    repo, _ = open_store(tmp_path, embedder.dim, projects=("api",))
    with pytest.raises(sqlite3.IntegrityError):
        repo.add(Memory.create("ghost note", project="ghost"))


def test_sqlite_a_global_memory_satisfies_the_fk(tmp_path):
    """A global memory carries project='__global__' (a seeded sentinel row), so it
    satisfies memories.project -> projects(slug) even with no real project registered."""
    embedder = HashEmbedder()
    repo, _ = open_store(tmp_path, embedder.dim)  # only the __global__ sentinel exists
    rule = Memory.create("always confirm destructive ops", type="rule", scope="global")
    assert rule.project == "__global__"
    repo.add(rule, embedder.encode(rule.content))
    assert {m.id for m in repo.list_all()} == {rule.id}


def test_sqlite_hybrid_finds_exact_token(tmp_path):
    embedder = HashEmbedder()
    repo, _ = open_store(tmp_path, embedder.dim, projects=("api",))
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


def test_delete_and_purge(open_repo, embedder):
    repo = open_repo()
    one = _store(repo, embedder, "one", project="api")
    _store(repo, embedder, "two", project="api")

    assert repo.delete([one.id]) == 1
    assert {m.content for m in repo.list_all()} == {"two"}
    assert repo.delete_all() == 1
    assert repo.list_all() == []


def _supersede_inputs(repo, embedder):
    """A stored prior + a successor wired the way the use case does it."""
    prior = _store(repo, embedder, "auth model v1", project="api", topic_key="auth/model")
    successor = Memory.create("auth model v2", project="api", topic_key="auth/model")
    successor.supersedes = prior.id
    return prior, successor


def test_supersede_marks_prior_and_inserts_successor(open_repo, embedder):
    repo = open_repo()
    prior, successor = _supersede_inputs(repo, embedder)

    repo.supersede(successor, embedder.encode(successor.content))

    active = repo.find_active_by_topic_key("auth/model", "api")
    assert active is not None and active.id == successor.id
    by_id = {m.id: m for m in repo.list_all()}
    assert by_id[prior.id].status == "superseded"
    assert by_id[successor.id].status == "active"
    assert by_id[successor.id].supersedes == prior.id  # the chain lives in this column


def test_supersede_is_atomic_on_failure(open_repo, embedder, monkeypatch):
    repo = open_repo()
    prior, successor = _supersede_inputs(repo, embedder)

    def boom(*args, **kwargs):
        raise RuntimeError("insert failed mid-supersede")

    monkeypatch.setattr(repo, "_insert_memory", boom)

    with pytest.raises(RuntimeError):
        repo.supersede(successor, embedder.encode(successor.content))

    # Nothing applied: the prior is still the active record and the successor was never
    # inserted — the mark-superseded + insert rolled back together.
    active = repo.find_active_by_topic_key("auth/model", "api")
    assert active is not None and active.id == prior.id
    assert {m.id for m in repo.list_all()} == {prior.id}
