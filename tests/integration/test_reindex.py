"""Reindex migration: switch embedders, rebuild at the new dimension, re-embed all."""
import json

import pytest

pytest.importorskip("sqlite_vec")

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.adapters.embedding.sync_embedding_scheduler import SyncEmbeddingScheduler
from mnemo.adapters.session.in_process_session_provider import InProcessSessionProvider
from mnemo.application.project_gate import ProjectGate
from mnemo.application.use_cases.reindex_memories import ReindexMemories
from mnemo.application.use_cases.remember_memory import RememberMemoryUseCaseImpl
from mnemo.application.use_cases.search_memory import SearchMemoryUseCaseImpl
from mnemo.domain.memory import Memory
from tests.support.sqlite_store import open_store


def _sqlite(tmp_path, dim):
    """A real SQLite store + its registry (with "api" registered for the gate + FK)."""
    return open_store(tmp_path, dim, projects=("api",))


@pytest.fixture
def store(tmp_path):
    return _sqlite(tmp_path, dim=256)


def _remember(store, embedder, content, **kwargs):
    repo, registry = store
    use_case = RememberMemoryUseCaseImpl(
        repo, SyncEmbeddingScheduler(embedder, repo), embedder,
        InProcessSessionProvider(), ProjectGate(registry),
    )
    return use_case.execute(content=content, project="api", **kwargs)


def _search(store, embedder):
    repo, registry = store
    return SearchMemoryUseCaseImpl(repo, embedder, ProjectGate(registry))


def _reindex(store, embedder):
    repo, _ = store
    return ReindexMemories(repo, embedder, SyncEmbeddingScheduler(embedder, repo)).execute()


def test_reindex_switches_dimension_and_reembeds_all(store):
    repo, _ = store
    old = HashEmbedder(dim=256)
    # a supersede so there is also a link to preserve
    first = _remember(store, old, "auth model v1", topic_key="auth/model")
    second = _remember(store, old, "auth model v2", topic_key="auth/model")
    other = _remember(store, old, "redis cache eviction")
    assert repo.pending_count() == 0  # embedded at dim 256

    new = HashEmbedder(dim=128)
    count = _reindex(store, new)

    assert count == 3  # all memories re-embedded — superseded history included
    assert repo.pending_count() == 0
    # content + metadata preserved
    by_id = {memory.id: memory for memory in repo.list_all()}
    assert by_id[other.id].content == "redis cache eviction"
    assert by_id[first.id].topic_key == "auth/model"
    # the supersede link survived the rebuild
    links = repo.links_for(second.id)
    assert len(links) == 1 and links[0].target_id == first.id
    # search works at the new dimension (a 128-dim store would reject a 256-dim vector)
    hits = _search(store, new).execute(query="redis cache", scope="all")
    assert any(hit.id == other.id for hit in hits)


def test_reindex_is_noop_when_dimension_unchanged(store):
    repo, _ = store
    embedder = HashEmbedder(dim=256)
    _remember(store, embedder, "a note")
    assert _reindex(store, embedder) == 0  # same dim, nothing pending
    assert repo.pending_count() == 0


def test_reindex_pages_through_all_pending(store, monkeypatch):
    # Force several pages so the drain loop — not a single pre-read count — is exercised:
    # every pending row must be re-embedded, none missed at a page boundary.
    import mnemo.application.use_cases.reindex_memories as reindex_module

    monkeypatch.setattr(reindex_module, "_REINDEX_PAGE", 2)
    repo, _ = store
    old = HashEmbedder(dim=256)
    for i in range(5):
        _remember(store, old, f"note {i}")
    assert repo.pending_count() == 0  # embedded at 256

    count = _reindex(store, HashEmbedder(dim=128))
    assert count == 5  # all five scheduled across pages
    assert repo.pending_count() == 0


def test_sqlite_reindex_rebuilds_schema_dim(tmp_path):
    store = _sqlite(tmp_path, dim=256)
    repo, _ = store
    _remember(store, HashEmbedder(dim=256), "note one")
    assert repo._current_dim() == 256

    _reindex(store, HashEmbedder(dim=128))
    assert repo._current_dim() == 128  # the embedding-column CHECK was rebuilt


def test_sqlite_set_dimension_is_atomic_on_failure(tmp_path, monkeypatch):
    """A failure mid-rebuild must leave the original store fully intact — never the
    0-rows-on-disk outcome of a non-atomic drop-then-recreate."""
    store = _sqlite(tmp_path, dim=256)
    repo, _ = store
    embedder = HashEmbedder(dim=256)
    kept = []
    for content in ["alpha note", "beta note", "gamma note"]:
        memory = Memory.create(content, project="api")
        repo.add(memory, embedder.encode(content))
        kept.append(memory)

    # Fail at the final rebuild step — AFTER the destructive drop/rename — so only an
    # atomic swap survives it; a drop-then-recreate would already have wiped the rows.
    def boom(*args, **kwargs):
        raise RuntimeError("injected mid-rebuild failure")

    monkeypatch.setattr(repo, "_rebuild_indexes_and_fts", boom)

    with pytest.raises(RuntimeError):
        repo.set_dimension(128)

    assert repo._current_dim() == 256  # dimension unchanged — the swap rolled back
    assert {m.id for m in repo.list_all()} == {m.id for m in kept}  # no rows lost
    # still fully queryable: the original table + FTS survived the rollback
    hits = _search(store, embedder).execute(query="beta note", scope="all")
    assert any(hit.id == kept[1].id for hit in hits)


def test_sqlite_set_dimension_rebuilds_fts(tmp_path):
    """After a swap the external-content FTS index is rebuilt. Proven via a STILL-PENDING
    memory (no vector → absent from the dense leg), so the only way to find it is the
    lexical leg reading the rebuilt FTS index."""
    store = _sqlite(tmp_path, dim=256)
    repo, _ = store
    _remember(store, HashEmbedder(dim=256), "a note with the rare token zzqqx inside")

    repo.set_dimension(128)  # everything back to pending; FTS rebuilt against the swap
    assert repo.pending_count() == 1  # not re-embedded → dense leg cannot surface it

    hits = _search(store, HashEmbedder(dim=128)).execute(query="zzqqx", scope="all")
    assert any("zzqqx" in hit.content for hit in hits)


def test_cli_reindex(tmp_path, monkeypatch):
    testing = pytest.importorskip("typer.testing")
    monkeypatch.setenv("MNEMO_EMBEDDER", "hash")
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))
    from mnemo.adapters.cli.app import app

    runner = testing.CliRunner()
    runner.invoke(app, ["store", "a memory to reindex", "--project", "api"])

    dry = runner.invoke(app, ["reindex", "--dry-run"])
    assert dry.exit_code == 0 and json.loads(dry.stdout)["dry_run"] is True

    run = runner.invoke(app, ["reindex"])
    assert run.exit_code == 0
    result = json.loads(run.stdout)
    assert "dim" in result
    assert result["service_restarted"] is False  # no shared service runs in the isolated env
