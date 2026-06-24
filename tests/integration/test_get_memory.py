"""get: dereference one memory by id or topic_key — the full record + its supersede chain."""
from __future__ import annotations

import pytest

pytest.importorskip("sqlite_vec")

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.adapters.embedding.sync_embedding_scheduler import SyncEmbeddingScheduler
from mnemo.adapters.session.in_process_session_provider import InProcessSessionProvider
from mnemo.application.project_gate import ProjectGate
from mnemo.application.use_cases.get_memory import GetMemoryUseCaseImpl
from mnemo.application.use_cases.remember_memory import RememberMemoryUseCaseImpl
from tests.support.sqlite_store import open_store


def _setup(tmp_path):
    repo, registry = open_store(tmp_path, dim=8, projects=("api",))
    embedder = HashEmbedder(dim=8)
    remember = RememberMemoryUseCaseImpl(
        repo, SyncEmbeddingScheduler(embedder, repo), embedder,
        InProcessSessionProvider(), ProjectGate(registry),
    )
    get = GetMemoryUseCaseImpl(repo, ProjectGate(registry))
    return remember, get


def test_get_by_topic_key_returns_the_active_head_and_full_chain(tmp_path):
    remember, get = _setup(tmp_path)
    v1 = remember.execute(content="auth v1", project="api", topic_key="auth/model")
    v2 = remember.execute(content="auth v2", project="api", topic_key="auth/model")
    v3 = remember.execute(content="auth v3", project="api", topic_key="auth/model")

    result = get.execute(topic_key="auth/model", project="api")
    assert result.id == v3.id and result.content == "auth v3"      # the active head
    assert result.status == "active" and result.topic_key == "auth/model"
    assert [e.id for e in result.chain] == [v3.id, v2.id, v1.id]   # newest -> oldest lineage
    assert [e.status for e in result.chain] == ["active", "superseded", "superseded"]
    assert result.chain_total == 3


def test_get_by_id_resolves_a_superseded_version_with_its_lineage(tmp_path):
    remember, get = _setup(tmp_path)
    v1 = remember.execute(content="auth v1", project="api", topic_key="auth/model")
    remember.execute(content="auth v2", project="api", topic_key="auth/model")

    result = get.execute(id=v1.id)  # a now-superseded version, by global id (no scope/project)
    assert result.id == v1.id and result.content == "auth v1"
    assert result.status == "superseded"
    assert result.chain_total == 2  # still sees the whole lineage from the head


def test_get_chain_limit_and_cursor_page_the_lineage(tmp_path):
    remember, get = _setup(tmp_path)
    ids = [remember.execute(content=f"v{i}", project="api", topic_key="t").id for i in range(5)]

    page = get.execute(topic_key="t", project="api", chain_limit=2)
    assert [e.id for e in page.chain] == [ids[4], ids[3]]  # the two newest
    assert page.chain_total == 5

    older = get.execute(topic_key="t", project="api", chain_limit=2, chain_after=ids[3])
    assert [e.id for e in older.chain] == [ids[2], ids[1]]  # the next-older window after ids[3]


def test_get_a_memory_without_a_topic_key_has_a_singleton_chain(tmp_path):
    remember, get = _setup(tmp_path)
    m = remember.execute(content="a one-off note", project="api")
    result = get.execute(id=m.id)
    assert result.topic_key is None
    assert [e.id for e in result.chain] == [m.id] and result.chain_total == 1


def test_get_by_topic_key_in_global_scope(tmp_path):
    remember, get = _setup(tmp_path)
    remember.execute(content="a global rule", scope="global", type="rule", topic_key="rule/x")
    result = get.execute(topic_key="rule/x", scope="global")
    assert result.scope == "global" and result.content == "a global rule"


def test_get_missing_topic_key_errors_with_near_match(tmp_path):
    remember, get = _setup(tmp_path)
    remember.execute(content="auth v1", project="api", topic_key="auth/model")
    with pytest.raises(ValueError) as exc:
        get.execute(topic_key="auth/modal", project="api")  # a typo
    message = str(exc.value)
    assert "auth/modal" in message and "auth/model" in message  # names the miss + suggests the real key


def test_get_missing_id_errors(tmp_path):
    _, get = _setup(tmp_path)
    with pytest.raises(ValueError, match="no memory with id"):
        get.execute(id="does-not-exist")


def test_get_requires_exactly_one_of_id_or_topic_key(tmp_path):
    _, get = _setup(tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        get.execute(id="x", topic_key="y")  # both
    with pytest.raises(ValueError, match="exactly one"):
        get.execute()  # neither
