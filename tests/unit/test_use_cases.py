import pytest

pytest.importorskip("sqlite_vec")

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.adapters.embedding.sync_embedding_scheduler import SyncEmbeddingScheduler
from mnemo.adapters.session.in_process_session_provider import InProcessSessionProvider
from mnemo.application.project_gate import ProjectGate, UnknownProject
from mnemo.application.use_cases.browse_memory import BrowseMemoryUseCaseImpl
from mnemo.application.use_cases.delete_memory import DeleteMemoryUseCaseImpl
from mnemo.application.use_cases.delete_project import DeleteProjectUseCaseImpl
from mnemo.application.use_cases.remember_memory import RememberMemoryUseCaseImpl
from mnemo.application.use_cases.search_memory import SearchMemoryUseCaseImpl
from mnemo.domain.project import Project
from tests.support.sqlite_store import open_store

_TEST_PROJECTS = ("api", "other", "svc-a", "svc-b")


def _build_wiring(tmp_path):
    """(repo, remember, search, delete, delete_project, browse, projects) over a real
    temp-file SQLite store with the test projects registered (the gate + FK need them)."""
    repo, projects = open_store(tmp_path, HashEmbedder().dim, projects=_TEST_PROJECTS)
    embedder = HashEmbedder()
    gate = ProjectGate(projects)
    session = InProcessSessionProvider()
    scheduler = SyncEmbeddingScheduler(embedder, repo)
    return (
        repo,
        RememberMemoryUseCaseImpl(repo, scheduler, embedder, session, gate),
        SearchMemoryUseCaseImpl(repo, embedder, gate),
        DeleteMemoryUseCaseImpl(repo, projects),
        DeleteProjectUseCaseImpl(projects),
        BrowseMemoryUseCaseImpl(repo, gate),
        projects,
    )


@pytest.fixture
def wiring(tmp_path):
    return _build_wiring(tmp_path)


def _registry(tmp_path):
    """Just the registry (real SQLite) for the project use-case unit tests."""
    _, projects = open_store(tmp_path, HashEmbedder().dim)
    return projects


def test_remember_then_search_finds_it(wiring):
    _, remember, search, *_ = wiring
    stored = remember.execute(
        content="Use JWT with refresh token rotation", type="decision", project="api"
    )
    assert stored.status == "created"
    hits = search.execute(query="jwt refresh rotation", project="api")
    assert any(hit.id == stored.id for hit in hits)


def test_remember_project_scope_without_a_project_is_rejected(wiring):
    # The write path enforces the same scope↔project contract the read path does: a
    # project-scoped write with no project would be unreachable by a project search.
    _, remember, *_ = wiring
    with pytest.raises(ValueError):
        remember.execute(content="orphan note")  # scope defaults to 'project', no project


def test_remember_global_scope_rejects_a_project(wiring):
    _, remember, *_ = wiring
    with pytest.raises(ValueError):
        remember.execute(content="a rule", scope="global", project="api")  # contradictory


def test_remember_unknown_project_is_rejected_with_candidates(wiring):
    _, remember, *_ = wiring
    with pytest.raises(UnknownProject) as exc:
        remember.execute(content="x", project="ap")  # a typo of the registered "api"
    assert "api" in exc.value.candidates


def test_project_scoped_search_without_a_project_errors(wiring):
    # scope defaults to 'project'; with no project there is nothing to scope to,
    # so the search fails fast instead of silently returning nothing.
    _, _, search, *_ = wiring
    with pytest.raises(ValueError):
        search.execute(query="anything")


def test_search_unknown_project_is_rejected(wiring):
    _, _, search, *_ = wiring
    with pytest.raises(UnknownProject):
        search.execute(query="anything", project="nope")


def test_browse_lists_newest_first_without_a_query(wiring):
    _, remember, _, _, _, browse, _ = wiring
    a = remember.execute(content="alpha", project="api")
    b = remember.execute(content="beta", project="api")

    results = browse.execute(project="api")
    created = [r.created_at for r in results]
    assert created == sorted(created, reverse=True)  # newest first
    assert {r.id for r in results} == {a.id, b.id}
    assert not hasattr(results[0], "score")  # browse hits carry no relevance score


def test_browse_inherits_the_scope_project_guard(wiring):
    *_, browse, _ = wiring
    with pytest.raises(ValueError):
        browse.execute()  # scope defaults to 'project' with no project


def test_sync_remember_embeds_immediately(wiring):
    # With the sync scheduler, a write ends fully embedded (no pending vector) —
    # same observable result as before deferred embedding.
    repo, remember, *_ = wiring
    stored = remember.execute(content="embed me now", project="api")
    assert repo.has_vector(stored.id) is True
    assert repo.pending_count() == 0


def test_exact_duplicate_is_not_stored_twice(wiring):
    repo, remember, *_ = wiring
    first = remember.execute(content="same content", project="api")
    second = remember.execute(content="  Same   Content ", project="api")
    assert second.status == "duplicate"
    assert second.id == first.id
    assert len(repo.list_all()) == 1


def test_exact_duplicate_under_a_new_topic_key_is_rejected(wiring):
    # The exact-dup guard runs before the topic_key upsert, so re-storing identical content
    # under a NEW topic_key would silently drop the intended evolution — it must be loud.
    repo, remember, *_ = wiring
    first = remember.execute(content="auth uses jwt", project="api")
    with pytest.raises(ValueError, match=first.id):  # the error names the existing memory
        remember.execute(content="auth uses jwt", project="api", topic_key="auth/model")
    assert len(repo.list_all()) == 1  # nothing was stored


def test_re_remembering_identical_content_under_the_same_topic_key_is_a_duplicate(wiring):
    # Same content AND the same topic_key is an idempotent re-store, not a re-key → soft dup.
    repo, remember, *_ = wiring
    first = remember.execute(content="auth uses jwt", project="api", topic_key="auth/model")
    second = remember.execute(content="auth uses jwt", project="api", topic_key="auth/model")
    assert second.status == "duplicate"
    assert second.id == first.id
    assert len(repo.list_all()) == 1


def test_re_keying_identical_content_to_another_topic_key_is_rejected(wiring):
    # Both keys set but different — moving identical content from key A to key B is the
    # deferred edit op, not a silent dup; it must also be a loud error.
    repo, remember, *_ = wiring
    remember.execute(content="auth uses jwt", project="api", topic_key="auth/v1")
    with pytest.raises(ValueError, match="auth/v1"):  # names the existing key
        remember.execute(content="auth uses jwt", project="api", topic_key="auth/v2")
    assert len(repo.list_all()) == 1


def test_same_content_in_two_projects_is_kept_separately(wiring):
    # The exact-dup hash key is global, but content is unique only within a scope: the
    # same fact in another project is a distinct memory, not a duplicate to drop.
    repo, remember, *_ = wiring
    first = remember.execute(content="shared fact", project="api")
    second = remember.execute(content="shared fact", project="other")
    assert second.status == "created"
    assert second.id != first.id
    assert len(repo.list_all()) == 2


def test_re_remembering_superseded_content_creates_a_fresh_row(wiring):
    # Re-storing content that was superseded must write a new, retrievable row — not
    # return the dead (superseded) id and silently store nothing.
    _, remember, search, *_ = wiring
    first = remember.execute(content="reborn note", project="api", topic_key="reborn")
    # A topic_key upsert supersedes `first` — the production path to a superseded row.
    remember.execute(content="a newer take", project="api", topic_key="reborn")

    second = remember.execute(content="reborn note", project="api")
    assert second.status == "created"
    assert second.id != first.id

    ids = {hit.id for hit in search.execute(query="reborn note", project="api")}
    assert second.id in ids and first.id not in ids


def test_topic_key_upsert_supersedes_prior(wiring):
    repo, remember, search, *_ = wiring
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


def test_topic_key_upsert_records_the_supersedes_pointer(wiring):
    repo, remember, *_ = wiring
    first = remember.execute(content="Auth model v1", project="api", topic_key="auth/model")
    second = remember.execute(content="Auth model v2", project="api", topic_key="auth/model")

    by_id = {m.id: m for m in repo.list_all()}
    assert by_id[second.id].supersedes == first.id  # the successor points at the prior

    # A first-time write (no prior topic_key) records no supersede pointer.
    solo = remember.execute(content="standalone note", project="api")
    assert next(m for m in repo.list_all() if m.id == solo.id).supersedes is None


def test_near_similar_memories_coexist(wiring):
    repo, remember, *_ = wiring
    a = remember.execute(content="postgres connection pool", type="decision", project="api")
    b = remember.execute(
        content="postgres connection pool tuning notes", type="decision", project="api"
    )
    assert a.id != b.id
    assert len(repo.list_all()) == 2


def test_default_project_scope_includes_global(wiring):
    _, remember, search, *_ = wiring
    rule = remember.execute(
        content="always confirm destructive operations", type="rule", scope="global"
    )
    remember.execute(content="checkout specific note", project="api")
    hits = search.execute(query="destructive operations", project="api")
    assert any(hit.id == rule.id for hit in hits)


def test_cross_project_search_with_scope_all(wiring):
    _, remember, search, *_ = wiring
    a = remember.execute(content="postgres connection pool limits", project="svc-a")
    b = remember.execute(content="postgres connection pool tuning", project="svc-b")
    ids = {hit.id for hit in search.execute(query="connection pool", scope="all")}
    assert {a.id, b.id} <= ids


def test_search_filters_by_tag(wiring):
    _, remember, search, *_ = wiring
    auth = remember.execute(content="jwt rotation", project="api", tags=["auth"])
    remember.execute(content="redis cache layer", project="api", tags=["cache"])
    hits = search.execute(query="jwt rotation", project="api", tags=["auth"])
    assert [hit.id for hit in hits] == [auth.id]


def test_search_created_after_keeps_fresh(wiring):
    _, remember, search, *_ = wiring
    fresh = remember.execute(content="fresh decision today", project="api")
    hits = search.execute(
        query="fresh decision today", project="api", created_after="2000-01-01"
    )
    assert any(hit.id == fresh.id for hit in hits)


def test_remember_stamps_one_session_id_per_run(wiring):
    repo, remember, *_ = wiring
    first = remember.execute(content="one", project="api")
    second = remember.execute(content="two", project="api")

    session_by_id = {memory.id: memory.session_id for memory in repo.list_all()}
    assert session_by_id[first.id] is not None
    assert session_by_id[first.id] == session_by_id[second.id]  # same run → same session


def test_distinct_runs_get_distinct_session_ids(tmp_path):
    repo_a, remember_a, *_ = _build_wiring(tmp_path / "a")
    repo_b, remember_b, *_ = _build_wiring(tmp_path / "b")
    a = remember_a.execute(content="note", project="api")
    b = remember_b.execute(content="note", project="api")

    session_a = {m.id: m.session_id for m in repo_a.list_all()}[a.id]
    session_b = {m.id: m.session_id for m in repo_b.list_all()}[b.id]
    assert session_a != session_b


def _remember_with_window(tmp_path, max_input, max_content=100_000):
    """A standalone remember wired with its own small-window embedder + a registry
    holding 'api' (the gate needs a registered project). max_content defaults large so a
    test exercises the embedder WINDOW unless it sets a smaller policy cap."""
    embedder = HashEmbedder(max_input=max_input)
    repo, projects = open_store(tmp_path, embedder.dim, projects=("api",))
    remember = RememberMemoryUseCaseImpl(
        repo, SyncEmbeddingScheduler(embedder, repo), embedder,
        InProcessSessionProvider(), ProjectGate(projects),
        max_content_tokens=max_content,
    )
    return repo, remember


def test_over_window_content_is_rejected_not_truncated(tmp_path):
    repo, remember = _remember_with_window(tmp_path, 5)  # window of 5 tokens
    with pytest.raises(ValueError):
        remember.execute(content="one two three four five six seven", project="api")
    assert repo.list_all() == []  # nothing stored on reject


def test_within_window_content_is_stored(tmp_path):
    repo, remember = _remember_with_window(tmp_path, 5)
    stored = remember.execute(content="one two three", project="api")
    assert stored.id
    assert len(repo.list_all()) == 1


def test_policy_cap_rejects_content_that_fits_the_embedder_window(tmp_path):
    # The embedder window is generous (100) but the per-memory policy cap (5) is stricter
    # and bites first — keeping memories focused even when the embedder could take more.
    repo, remember = _remember_with_window(tmp_path, 100, max_content=5)
    with pytest.raises(ValueError):
        remember.execute(content="one two three four five six seven", project="api")
    assert repo.list_all() == []


def test_delete_project_removes_it_from_the_registry(wiring):
    *_, delete_project, _, projects = wiring
    deleted = delete_project.execute("api")
    assert deleted.slug == "api"
    assert projects.exists("api") is False  # cascade of its memories is a store-level concern


def test_delete_project_unknown_is_rejected_with_candidates(wiring):
    *_, delete_project, _, _ = wiring
    with pytest.raises(UnknownProject) as exc:
        delete_project.execute("ap")  # a typo of the registered "api"
    assert "api" in exc.value.candidates


def test_update_project_sets_description(tmp_path):
    from mnemo.application.use_cases.update_project import UpdateProjectUseCaseImpl

    projects = _registry(tmp_path)
    projects.create(Project.create("api"))
    updated = UpdateProjectUseCaseImpl(projects).execute("api", "the api service")
    assert updated.description == "the api service"
    assert projects.get("api").description == "the api service"


def test_update_project_unknown_is_rejected_with_candidates(tmp_path):
    from mnemo.application.use_cases.update_project import UpdateProjectUseCaseImpl

    projects = _registry(tmp_path)
    projects.create(Project.create("api"))
    with pytest.raises(UnknownProject) as exc:
        UpdateProjectUseCaseImpl(projects).execute("ap", "x")  # typo of "api"
    assert "api" in exc.value.candidates


def test_list_projects_lists_registered_excluding_global(tmp_path):
    from mnemo.application.use_cases.list_projects import ListProjectsUseCaseImpl

    projects = _registry(tmp_path)
    projects.create(Project.create("api"))
    projects.create(Project.create("svc"))
    slugs = {p.slug for p in ListProjectsUseCaseImpl(projects).execute()}
    assert slugs == {"api", "svc"}  # the __global__ sentinel is hidden


def test_delete_and_purge(wiring):
    repo, remember, _, deletion, *_ = wiring
    a = remember.execute(content="one", project="api")
    remember.execute(content="two", project="api")

    assert deletion.delete([a.id]).deleted == 1
    assert {m.content for m in repo.list_all()} == {"two"}
    assert deletion.purge().deleted == 1
    assert repo.list_all() == []


def test_purge_wipes_memories_and_projects(wiring):
    repo, remember, _, deletion, _, _, projects = wiring
    remember.execute(content="one", project="api")

    deletion.purge()

    assert repo.list_all() == []
    assert projects.list_all() == []        # the registry is wiped too (full reset)
    assert projects.exists("api") is False  # ... including previously-registered projects


def test_deleting_active_head_then_re_remembering_evolves_not_forks(wiring):
    repo, remember, _, deletion, *_ = wiring
    remember.execute(content="auth v1", project="api", topic_key="auth/model")
    head = remember.execute(content="auth v2", project="api", topic_key="auth/model")

    deletion.delete([head.id])  # delete the active head of the chain

    third = remember.execute(content="auth v3", project="api", topic_key="auth/model")
    assert third.status == "superseded"  # should evolve the promoted prior, not fork
