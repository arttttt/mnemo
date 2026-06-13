from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.adapters.store.in_memory_repository import InMemoryMemoryRepository
from mnemo.application.use_cases.delete_memory import DeleteMemory
from mnemo.application.use_cases.remember_memory import RememberMemory
from mnemo.application.use_cases.search_memory import SearchMemory


def _wiring():
    repo = InMemoryMemoryRepository()
    embedder = HashEmbedder()
    return (
        repo,
        RememberMemory(repo, embedder),
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
