"""Reindex migration: switch embedders, rebuild at the new dimension, re-embed all."""
import json

import pytest

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.adapters.embedding.sync_embedding_scheduler import SyncEmbeddingScheduler
from mnemo.adapters.session.in_process_session_provider import InProcessSessionProvider
from mnemo.application.use_cases.reindex_memories import ReindexMemories
from mnemo.application.use_cases.remember_memory import RememberMemory
from mnemo.application.use_cases.search_memory import SearchMemory


def _remember(repo, embedder, content, **kwargs):
    use_case = RememberMemory(
        repo, SyncEmbeddingScheduler(embedder, repo), InProcessSessionProvider()
    )
    return use_case.execute(content=content, project="api", **kwargs)


def _reindex(repo, embedder):
    return ReindexMemories(repo, embedder, SyncEmbeddingScheduler(embedder, repo)).execute()


def _in_memory(tmp_path):
    from mnemo.adapters.store.in_memory_repository import InMemoryMemoryRepository

    return InMemoryMemoryRepository(path=str(tmp_path / "memory.json"))


def _sqlite(tmp_path, dim):
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.sqlite_vec_repository import SqliteVecMemoryRepository

    return SqliteVecMemoryRepository(path=str(tmp_path / "memory.db"), dim=dim)


@pytest.fixture(params=["in_memory", "sqlite"])
def repo(request, tmp_path):
    return _in_memory(tmp_path) if request.param == "in_memory" else _sqlite(tmp_path, dim=256)


def test_reindex_switches_dimension_and_reembeds_all(repo):
    old = HashEmbedder(dim=256)
    # a supersede so there is also a link to preserve
    first = _remember(repo, old, "auth model v1", topic_key="auth/model")
    second = _remember(repo, old, "auth model v2", topic_key="auth/model")
    other = _remember(repo, old, "redis cache eviction")
    assert repo.pending_count() == 0  # embedded at dim 256

    new = HashEmbedder(dim=128)
    count = _reindex(repo, new)

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
    hits = SearchMemory(repo, new).execute(query="redis cache", scope="all")
    assert any(hit.id == other.id for hit in hits)


def test_reindex_is_noop_when_dimension_unchanged(repo):
    embedder = HashEmbedder(dim=256)
    _remember(repo, embedder, "a note")
    assert _reindex(repo, embedder) == 0  # same dim, nothing pending
    assert repo.pending_count() == 0


def test_sqlite_reindex_rebuilds_schema_dim(tmp_path):
    repo = _sqlite(tmp_path, dim=256)
    _remember(repo, HashEmbedder(dim=256), "note one")
    assert repo._current_dim() == 256

    _reindex(repo, HashEmbedder(dim=128))
    assert repo._current_dim() == 128  # the embedding-column CHECK was rebuilt


def test_cli_reindex(tmp_path, monkeypatch):
    testing = pytest.importorskip("typer.testing")
    monkeypatch.setenv("MNEMO_EMBEDDER", "hash")
    monkeypatch.setenv("MNEMO_STORE", "memory")
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MNEMO_STORE_PATH", str(tmp_path / "memory.json"))
    from mnemo.adapters.cli.app import app

    runner = testing.CliRunner()
    runner.invoke(app, ["store", "a memory to reindex", "--project", "api"])

    dry = runner.invoke(app, ["reindex", "--dry-run"])
    assert dry.exit_code == 0 and json.loads(dry.stdout)["dry_run"] is True

    run = runner.invoke(app, ["reindex"])
    assert run.exit_code == 0
    assert "dim" in json.loads(run.stdout)
