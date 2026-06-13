from mnemo.adapters.embedding.hashing import HashEmbedder
from mnemo.adapters.store.in_memory import InMemoryMemoryRepository
from mnemo.application.use_cases import RememberMemory, SearchMemory


def _wiring():
    repo = InMemoryMemoryRepository()
    embedder = HashEmbedder()
    return repo, RememberMemory(repo, embedder), SearchMemory(repo, embedder)


def test_remember_then_search_finds_it():
    _, remember, search = _wiring()
    stored = remember.execute(
        content="Use JWT with refresh token rotation", type="decision", project="api"
    )
    assert stored.dedup is None
    hits = search.execute(query="jwt refresh rotation", project="api")
    assert any(hit.id == stored.id for hit in hits)


def test_exact_duplicate_is_not_stored_twice():
    repo, remember, _ = _wiring()
    first = remember.execute(content="same content", project="api")
    second = remember.execute(content="  Same   Content ", project="api")
    assert second.dedup == "exact"
    assert second.id == first.id
    assert len(repo.list_all()) == 1


def test_near_duplicate_is_detected():
    _, remember, _ = _wiring()
    first = remember.execute(
        content="postgres connection pool", type="decision", project="api"
    )
    again = remember.execute(
        content="pool postgres connection", type="decision", project="api"
    )
    assert again.dedup == "near"
    assert again.id == first.id


def test_default_project_scope_includes_global():
    _, remember, search = _wiring()
    rule = remember.execute(
        content="always confirm destructive operations", type="rule", scope="global"
    )
    remember.execute(content="checkout specific note", project="api")
    hits = search.execute(query="destructive operations", project="api")
    assert any(hit.id == rule.id for hit in hits)


def test_cross_project_search_with_scope_all():
    _, remember, search = _wiring()
    a = remember.execute(content="postgres connection pool limits", project="svc-a")
    b = remember.execute(content="postgres connection pool tuning", project="svc-b")
    ids = {hit.id for hit in search.execute(query="connection pool", scope="all")}
    assert {a.id, b.id} <= ids


def test_project_scope_excludes_other_projects():
    _, remember, search = _wiring()
    remember.execute(content="alpha only detail", project="alpha")
    hits = search.execute(query="alpha only detail", project="beta")
    assert all(hit.project != "alpha" for hit in hits)
