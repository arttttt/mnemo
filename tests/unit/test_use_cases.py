import pytest

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.adapters.embedding.sync_embedding_scheduler import SyncEmbeddingScheduler
from mnemo.adapters.session.in_process_session_provider import InProcessSessionProvider
from tests.fakes.in_memory_project_repository import InMemoryProjectRepositoryImpl
from tests.fakes.in_memory_repository import InMemoryRepositoryImpl
from mnemo.application.project_gate import ProjectGate, UnknownProject
from mnemo.application.use_cases.browse_memory import BrowseMemoryUseCaseImpl
from mnemo.application.use_cases.delete_memory import DeleteMemoryUseCaseImpl
from mnemo.application.use_cases.delete_project import DeleteProjectUseCaseImpl
from mnemo.application.use_cases.remember_memory import RememberMemoryUseCaseImpl
from mnemo.application.use_cases.search_memory import SearchMemoryUseCaseImpl
from mnemo.domain.project import Project

_TEST_PROJECTS = ("api", "other", "svc-a", "svc-b")


def _wiring():
    """(repo, remember, search, delete, delete_project, browse, projects). The registry
    is pre-seeded with the projects these tests use, since the gate now requires
    registered projects."""
    repo = InMemoryRepositoryImpl()
    embedder = HashEmbedder()
    projects = InMemoryProjectRepositoryImpl()
    for slug in _TEST_PROJECTS:
        projects.create(Project.create(slug))
    gate = ProjectGate(projects)
    session = InProcessSessionProvider()
    scheduler = SyncEmbeddingScheduler(embedder, repo)
    return (
        repo,
        RememberMemoryUseCaseImpl(repo, scheduler, embedder, session, gate),
        SearchMemoryUseCaseImpl(repo, embedder, gate),
        DeleteMemoryUseCaseImpl(repo),
        DeleteProjectUseCaseImpl(projects),
        BrowseMemoryUseCaseImpl(repo, gate),
        projects,
    )


def test_remember_then_search_finds_it():
    _, remember, search, *_ = _wiring()
    stored = remember.execute(
        content="Use JWT with refresh token rotation", type="decision", project="api"
    )
    assert stored.status == "created"
    hits = search.execute(query="jwt refresh rotation", project="api")
    assert any(hit.id == stored.id for hit in hits)


def test_remember_project_scope_without_a_project_is_rejected():
    # The write path enforces the same scope↔project contract the read path does: a
    # project-scoped write with no project would be unreachable by a project search.
    _, remember, *_ = _wiring()
    with pytest.raises(ValueError):
        remember.execute(content="orphan note")  # scope defaults to 'project', no project


def test_remember_global_scope_rejects_a_project():
    _, remember, *_ = _wiring()
    with pytest.raises(ValueError):
        remember.execute(content="a rule", scope="global", project="api")  # contradictory


def test_remember_unknown_project_is_rejected_with_candidates():
    _, remember, *_ = _wiring()
    with pytest.raises(UnknownProject) as exc:
        remember.execute(content="x", project="ap")  # a typo of the registered "api"
    assert "api" in exc.value.candidates


def test_project_scoped_search_without_a_project_errors():
    # scope defaults to 'project'; with no project there is nothing to scope to,
    # so the search fails fast instead of silently returning nothing.
    _, _, search, *_ = _wiring()
    with pytest.raises(ValueError):
        search.execute(query="anything")


def test_search_unknown_project_is_rejected():
    _, _, search, *_ = _wiring()
    with pytest.raises(UnknownProject):
        search.execute(query="anything", project="nope")


def test_browse_lists_newest_first_without_a_query():
    _, remember, _, _, _, browse, _ = _wiring()
    a = remember.execute(content="alpha", project="api")
    b = remember.execute(content="beta", project="api")

    results = browse.execute(project="api")
    created = [r.created_at for r in results]
    assert created == sorted(created, reverse=True)  # newest first
    assert {r.id for r in results} == {a.id, b.id}
    assert not hasattr(results[0], "score")  # browse hits carry no relevance score


def test_browse_inherits_the_scope_project_guard():
    *_, browse, _ = _wiring()
    with pytest.raises(ValueError):
        browse.execute()  # scope defaults to 'project' with no project


def test_sync_remember_embeds_immediately():
    # With the sync scheduler, a write ends fully embedded (no pending vector) —
    # same observable result as before deferred embedding.
    repo, remember, *_ = _wiring()
    stored = remember.execute(content="embed me now", project="api")
    assert repo.has_vector(stored.id) is True
    assert repo.pending_count() == 0


def test_exact_duplicate_is_not_stored_twice():
    repo, remember, *_ = _wiring()
    first = remember.execute(content="same content", project="api")
    second = remember.execute(content="  Same   Content ", project="api")
    assert second.status == "duplicate"
    assert second.id == first.id
    assert len(repo.list_all()) == 1


def test_same_content_in_two_projects_is_kept_separately():
    # The exact-dup hash key is global, but content is unique only within a scope: the
    # same fact in another project is a distinct memory, not a duplicate to drop.
    repo, remember, *_ = _wiring()
    first = remember.execute(content="shared fact", project="api")
    second = remember.execute(content="shared fact", project="other")
    assert second.status == "created"
    assert second.id != first.id
    assert len(repo.list_all()) == 2


def test_re_remembering_superseded_content_creates_a_fresh_row():
    # Re-storing content that was superseded must write a new, retrievable row — not
    # return the dead (superseded) id and silently store nothing.
    _, remember, search, *_ = _wiring()
    first = remember.execute(content="reborn note", project="api", topic_key="reborn")
    # A topic_key upsert supersedes `first` — the production path to a superseded row.
    remember.execute(content="a newer take", project="api", topic_key="reborn")

    second = remember.execute(content="reborn note", project="api")
    assert second.status == "created"
    assert second.id != first.id

    ids = {hit.id for hit in search.execute(query="reborn note", project="api")}
    assert second.id in ids and first.id not in ids


def test_topic_key_upsert_supersedes_prior():
    repo, remember, search, *_ = _wiring()
    first = remember.execute(
        content="Auth model v1", type="decision", project="api", topic_key="auth/model"
    )
    second = remember.execute(
        content="Auth model v2", type="decision", project="api", topic_key="auth/model"
    )
    assert second.status == "superseded"  # which record it replaced lives in the links edge

    status_by_id = {m.id: m.status for m in repo.list_all()}
    assert status_by_id[first.id] == "superseded"
    assert status_by_id[second.id] == "active"

    ids = {hit.id for hit in search.execute(query="auth model", project="api")}
    assert second.id in ids and first.id not in ids


def test_topic_key_upsert_writes_a_supersedes_link():
    from mnemo.domain.link_type import LinkType

    repo, remember, *_ = _wiring()
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
    repo, remember, *_ = _wiring()
    a = remember.execute(content="postgres connection pool", type="decision", project="api")
    b = remember.execute(
        content="postgres connection pool tuning notes", type="decision", project="api"
    )
    assert a.id != b.id
    assert len(repo.list_all()) == 2


def test_default_project_scope_includes_global():
    _, remember, search, *_ = _wiring()
    rule = remember.execute(
        content="always confirm destructive operations", type="rule", scope="global"
    )
    remember.execute(content="checkout specific note", project="api")
    hits = search.execute(query="destructive operations", project="api")
    assert any(hit.id == rule.id for hit in hits)


def test_cross_project_search_with_scope_all():
    _, remember, search, *_ = _wiring()
    a = remember.execute(content="postgres connection pool limits", project="svc-a")
    b = remember.execute(content="postgres connection pool tuning", project="svc-b")
    ids = {hit.id for hit in search.execute(query="connection pool", scope="all")}
    assert {a.id, b.id} <= ids


def test_search_filters_by_tag():
    _, remember, search, *_ = _wiring()
    auth = remember.execute(content="jwt rotation", project="api", tags=["auth"])
    remember.execute(content="redis cache layer", project="api", tags=["cache"])
    hits = search.execute(query="jwt rotation", project="api", tags=["auth"])
    assert [hit.id for hit in hits] == [auth.id]


def test_search_created_after_keeps_fresh():
    _, remember, search, *_ = _wiring()
    fresh = remember.execute(content="fresh decision today", project="api")
    hits = search.execute(
        query="fresh decision today", project="api", created_after="2000-01-01"
    )
    assert any(hit.id == fresh.id for hit in hits)


def test_remember_stamps_one_session_id_per_run():
    repo, remember, *_ = _wiring()
    first = remember.execute(content="one", project="api")
    second = remember.execute(content="two", project="api")

    session_by_id = {memory.id: memory.session_id for memory in repo.list_all()}
    assert session_by_id[first.id] is not None
    assert session_by_id[first.id] == session_by_id[second.id]  # same run → same session


def test_distinct_runs_get_distinct_session_ids():
    repo_a, remember_a, *_ = _wiring()
    repo_b, remember_b, *_ = _wiring()
    a = remember_a.execute(content="note", project="api")
    b = remember_b.execute(content="note", project="api")

    session_a = {m.id: m.session_id for m in repo_a.list_all()}[a.id]
    session_b = {m.id: m.session_id for m in repo_b.list_all()}[b.id]
    assert session_a != session_b


def _remember_with_window(max_input):
    """A standalone remember wired with its own small-window embedder + a registry
    holding 'api' (the gate needs a registered project)."""
    repo = InMemoryRepositoryImpl()
    embedder = HashEmbedder(max_input=max_input)
    projects = InMemoryProjectRepositoryImpl()
    projects.create(Project.create("api"))
    remember = RememberMemoryUseCaseImpl(
        repo, SyncEmbeddingScheduler(embedder, repo), embedder,
        InProcessSessionProvider(), ProjectGate(projects),
    )
    return repo, remember


def test_over_window_content_is_rejected_not_truncated():
    repo, remember = _remember_with_window(5)  # window of 5 tokens
    with pytest.raises(ValueError):
        remember.execute(content="one two three four five six seven", project="api")
    assert repo.list_all() == []  # nothing stored on reject


def test_within_window_content_is_stored():
    repo, remember = _remember_with_window(5)
    stored = remember.execute(content="one two three", project="api")
    assert stored.id
    assert len(repo.list_all()) == 1


def test_delete_project_removes_it_from_the_registry():
    *_, delete_project, _, projects = _wiring()
    deleted = delete_project.execute("api")
    assert deleted.slug == "api"
    assert projects.exists("api") is False  # cascade of its memories is a store-level concern


def test_delete_project_unknown_is_rejected_with_candidates():
    *_, delete_project, _, _ = _wiring()
    with pytest.raises(UnknownProject) as exc:
        delete_project.execute("ap")  # a typo of the registered "api"
    assert "api" in exc.value.candidates


def test_update_project_sets_description():
    from mnemo.application.use_cases.update_project import UpdateProjectUseCaseImpl

    projects = InMemoryProjectRepositoryImpl()
    projects.create(Project.create("api"))
    updated = UpdateProjectUseCaseImpl(projects).execute("api", "the api service")
    assert updated.description == "the api service"
    assert projects.get("api").description == "the api service"


def test_update_project_unknown_is_rejected_with_candidates():
    from mnemo.application.use_cases.update_project import UpdateProjectUseCaseImpl

    projects = InMemoryProjectRepositoryImpl()
    projects.create(Project.create("api"))
    with pytest.raises(UnknownProject) as exc:
        UpdateProjectUseCaseImpl(projects).execute("ap", "x")  # typo of "api"
    assert "api" in exc.value.candidates


def test_list_projects_lists_registered_excluding_global():
    from mnemo.application.use_cases.list_projects import ListProjectsUseCaseImpl

    projects = InMemoryProjectRepositoryImpl()
    projects.create(Project.create("api"))
    projects.create(Project.create("svc"))
    slugs = {p.slug for p in ListProjectsUseCaseImpl(projects).execute()}
    assert slugs == {"api", "svc"}  # the __global__ sentinel is hidden


def test_delete_and_purge():
    repo, remember, _, deletion, *_ = _wiring()
    a = remember.execute(content="one", project="api")
    remember.execute(content="two", project="api")

    assert deletion.delete([a.id]).deleted == 1
    assert {m.content for m in repo.list_all()} == {"two"}
    assert deletion.purge().deleted == 1
    assert repo.list_all() == []
