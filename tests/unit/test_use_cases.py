import pytest

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.adapters.session.in_process_session_provider import InProcessSessionProvider
from mnemo.adapters.store.in_memory_repository import InMemoryMemoryRepository
from mnemo.application.use_cases.delete_memory import DeleteMemory
from mnemo.application.use_cases.remember_memory import RememberMemory
from mnemo.application.use_cases.search_memory import SearchMemory


def _wiring():
    repo = InMemoryMemoryRepository()
    embedder = HashEmbedder()
    return (
        repo,
        RememberMemory(repo, embedder, InProcessSessionProvider()),
        SearchMemory(repo, embedder),
        DeleteMemory(repo),
    )


def test_remember_then_search_finds_it():
    _, remember, search, _ = _wiring()
    stored = remember.execute(
        content="Use JWT with refresh token rotation", type="decision", project="api"
    )
    assert stored.dedup is None
    hits = search.execute(query="jwt refresh rotation", project="api")
    assert any(hit.id == stored.id for hit in hits)


def test_exact_duplicate_is_not_stored_twice():
    repo, remember, _, _ = _wiring()
    first = remember.execute(content="same content", project="api")
    second = remember.execute(content="  Same   Content ", project="api")
    assert second.dedup == "exact"
    assert second.id == first.id
    assert len(repo.list_all()) == 1


def test_topic_key_upsert_supersedes_prior():
    repo, remember, search, _ = _wiring()
    first = remember.execute(
        content="Auth model v1", type="decision", project="api", topic_key="auth/model"
    )
    second = remember.execute(
        content="Auth model v2", type="decision", project="api", topic_key="auth/model"
    )
    assert second.superseded == first.id

    status_by_id = {m.id: m.status for m in repo.list_all()}
    assert status_by_id[first.id] == "superseded"
    assert status_by_id[second.id] == "active"

    ids = {hit.id for hit in search.execute(query="auth model", project="api")}
    assert second.id in ids and first.id not in ids


def test_topic_key_upsert_writes_a_supersedes_link():
    from mnemo.domain.link_type import LinkType

    repo, remember, _, _ = _wiring()
    first = remember.execute(content="Auth model v1", project="api", topic_key="auth/model")
    second = remember.execute(content="Auth model v2", project="api", topic_key="auth/model")

    links = repo.links_for(second.id)
    assert len(links) == 1
    link = links[0]
    assert link.type is LinkType.SUPERSEDES
    assert (link.source_id, link.target_id) == (second.id, first.id)
    assert link.provenance == "auth/model"  # the topic_key that drove the upsert

    # A first-time write (no prior topic_key) creates no edge.
    solo = remember.execute(content="standalone note", project="api")
    assert repo.links_for(solo.id) == []


def test_near_similar_memories_coexist():
    repo, remember, _, _ = _wiring()
    a = remember.execute(content="postgres connection pool", type="decision", project="api")
    b = remember.execute(
        content="postgres connection pool tuning notes", type="decision", project="api"
    )
    assert a.id != b.id
    assert len(repo.list_all()) == 2


def test_default_project_scope_includes_global():
    _, remember, search, _ = _wiring()
    rule = remember.execute(
        content="always confirm destructive operations", type="rule", scope="global"
    )
    remember.execute(content="checkout specific note", project="api")
    hits = search.execute(query="destructive operations", project="api")
    assert any(hit.id == rule.id for hit in hits)


def test_cross_project_search_with_scope_all():
    _, remember, search, _ = _wiring()
    a = remember.execute(content="postgres connection pool limits", project="svc-a")
    b = remember.execute(content="postgres connection pool tuning", project="svc-b")
    ids = {hit.id for hit in search.execute(query="connection pool", scope="all")}
    assert {a.id, b.id} <= ids


def test_search_filters_by_tag():
    _, remember, search, _ = _wiring()
    auth = remember.execute(content="jwt rotation", project="api", tags=["auth"])
    remember.execute(content="redis cache layer", project="api", tags=["cache"])
    hits = search.execute(query="jwt rotation", project="api", tags=["auth"])
    assert [hit.id for hit in hits] == [auth.id]


def test_search_recency_days_keeps_fresh():
    _, remember, search, _ = _wiring()
    fresh = remember.execute(content="fresh decision today", project="api")
    hits = search.execute(query="fresh decision today", project="api", recency_days=7)
    assert any(hit.id == fresh.id for hit in hits)


def test_remember_stamps_one_session_id_per_run():
    repo, remember, _, _ = _wiring()
    first = remember.execute(content="one", project="api")
    second = remember.execute(content="two", project="api")

    session_by_id = {memory.id: memory.session_id for memory in repo.list_all()}
    assert session_by_id[first.id] is not None
    assert session_by_id[first.id] == session_by_id[second.id]  # same run → same session


def test_distinct_runs_get_distinct_session_ids():
    repo_a, remember_a, _, _ = _wiring()
    repo_b, remember_b, _, _ = _wiring()
    a = remember_a.execute(content="note", project="api")
    b = remember_b.execute(content="note", project="api")

    session_a = {m.id: m.session_id for m in repo_a.list_all()}[a.id]
    session_b = {m.id: m.session_id for m in repo_b.list_all()}[b.id]
    assert session_a != session_b


def test_over_window_content_is_rejected_not_truncated():
    repo = InMemoryMemoryRepository()
    embedder = HashEmbedder(max_input=5)  # window of 5 tokens
    remember = RememberMemory(repo, embedder, InProcessSessionProvider())
    with pytest.raises(ValueError):
        remember.execute(content="one two three four five six seven", project="api")
    assert repo.list_all() == []  # nothing stored on reject


def test_within_window_content_is_stored():
    repo = InMemoryMemoryRepository()
    embedder = HashEmbedder(max_input=5)
    remember = RememberMemory(repo, embedder, InProcessSessionProvider())
    stored = remember.execute(content="one two three", project="api")
    assert stored.id
    assert len(repo.list_all()) == 1


def test_delete_clear_purge():
    repo, remember, _, deletion = _wiring()
    a = remember.execute(content="one", project="api")
    remember.execute(content="two", project="api")
    remember.execute(content="three", project="other")

    assert deletion.delete([a.id]).deleted == 1
    assert deletion.clear("api").deleted == 1
    assert {m.project for m in repo.list_all()} == {"other"}
    assert deletion.purge().deleted == 1
    assert repo.list_all() == []
