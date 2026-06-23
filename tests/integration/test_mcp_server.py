import asyncio
import json

import pytest

pytest.importorskip("mcp")
pytest.importorskip("sqlite_vec")

from mnemo.adapters.mcp.server import build_mcp
from mnemo.application.use_cases.recall_project import RecallProjectUseCaseImpl
from mnemo.domain.memory_type import MemoryType
from mnemo.infrastructure.composition import build_container
from mnemo.infrastructure.config import Config


def _container(tmp_path):
    config = Config(
        data_dir=str(tmp_path),
        embedder="hash",
        sqlite_path=str(tmp_path / "memory.db"),
    )
    return build_container(config)


def _tools(tmp_path):
    mcp = build_mcp(_container(tmp_path))
    return {tool.name: tool for tool in asyncio.run(mcp.list_tools())}


def test_mcp_exposes_the_agent_tools(tmp_path):
    assert {
        "remember", "search", "browse", "recall", "delete", "purge",
        "create_project", "delete_project", "update_project", "list_projects",
    } <= set(_tools(tmp_path))


def test_remember_advertises_allowed_types(tmp_path):
    schema = _tools(tmp_path)["remember"].inputSchema
    type_schema = schema["properties"]["type"]
    enum = type_schema.get("enum")
    assert enum is not None, type_schema
    assert set(enum) == {member.value for member in MemoryType}


def test_optional_params_expose_concrete_types(tmp_path):
    # Optional[...] would render as anyOf[T, null] (some MCP clients show that as
    # "unknown"); every parameter must expose a concrete type or enum instead.
    for name, tool in _tools(tmp_path).items():
        for pname, pschema in (tool.inputSchema.get("properties") or {}).items():
            assert "anyOf" not in pschema, f"{name}.{pname} uses anyOf"
            assert "type" in pschema or "enum" in pschema, f"{name}.{pname} lacks a type"


def test_only_required_params_are_marked_required(tmp_path):
    required = {
        name: tool.inputSchema.get("required", [])
        for name, tool in _tools(tmp_path).items()
    }
    assert required["remember"] == ["content"]
    assert required["search"] == ["query"]
    assert required["browse"] == []  # query-less: every param is optional
    assert required["recall"] == ["query", "project"]
    assert required["delete"] == ["ids"]
    assert required["purge"] == []
    assert required["create_project"] == ["name"]
    assert required["delete_project"] == ["name"]
    assert required["update_project"] == ["name", "description"]
    assert required["list_projects"] == []


def _call(mcp, name, args):
    outcome = asyncio.run(mcp.call_tool(name, args))
    blocks = outcome[0] if isinstance(outcome, tuple) else outcome
    return [block.text for block in blocks]


def test_mcp_remember_search_and_delete_roundtrip(tmp_path):
    mcp = build_mcp(_container(tmp_path))
    _call(mcp, "create_project", {"name": "api"})
    stored = json.loads(
        _call(mcp, "remember", {"content": "jwt refresh rotation", "type": "decision", "project": "api"})[0]
    )

    found = _call(mcp, "search", {"query": "jwt rotation", "project": "api"})
    assert any(stored["id"] in hit for hit in found)

    assert json.loads(_call(mcp, "delete", {"ids": [stored["id"]]})[0])["deleted"] == 1
    assert _call(mcp, "search", {"query": "jwt rotation", "project": "api"}) == []


def test_mcp_remember_enforces_the_per_type_cap(tmp_path):
    # A `rule` is capped at 128 tokens, a `decision` at 512: the SAME ~140-token content is
    # rejected as a rule but stored as a decision — enforced end-to-end through the MCP tool.
    mcp = build_mcp(_container(tmp_path))
    _call(mcp, "create_project", {"name": "api"})
    content = " ".join(f"w{i}" for i in range(140))

    with pytest.raises(Exception, match=r"128-token limit for a 'rule'"):
        _call(mcp, "remember", {"content": content, "type": "rule", "project": "api"})

    stored = json.loads(
        _call(mcp, "remember", {"content": content, "type": "decision", "project": "api"})[0]
    )
    assert stored["status"] == "created"


class _StubGenerator:
    """A deterministic generator so recall is exercised end-to-end without a real model."""

    def generate(self, prompt, *, max_tokens):
        assert "jwt refresh rotation" in prompt  # the gathered memory reached the synthesis prompt
        return "auth uses jwt refresh rotation"


def test_mcp_recall_synthesizes_a_grounded_answer(tmp_path):
    container = _container(tmp_path)
    # Inject a stub generator (the default is the real GGUF model — too heavy for this test).
    container.recall = RecallProjectUseCaseImpl(
        container.repository, container.embedder, reranker=None, generator=_StubGenerator(),
        rerank_top_k=20, generator_max_tokens=128,
    )
    mcp = build_mcp(container)
    _call(mcp, "create_project", {"name": "api"})
    remembered = json.loads(
        _call(mcp, "remember", {"content": "jwt refresh rotation", "type": "decision", "project": "api"})[0]
    )

    result = json.loads(_call(mcp, "recall", {"query": "auth", "project": "api"})[0])

    assert result["project"] == "api"
    assert result["summary"] == "auth uses jwt refresh rotation"  # the synthesized answer
    # the supporting memory comes back only as a light reference — id + type, never its content
    assert result["sources"] == [{"id": remembered["id"], "type": "decision"}]
    assert "sections" not in result  # full memory content is not dumped to the caller


def test_mcp_delete_project_cascades(tmp_path):
    mcp = build_mcp(_container(tmp_path))
    _call(mcp, "create_project", {"name": "api"})
    _call(mcp, "remember", {"content": "doomed via mcp", "project": "api"})

    deleted = json.loads(_call(mcp, "delete_project", {"name": "api"})[0])
    assert deleted["slug"] == "api"
    # its memory cascaded away — a cross-project search no longer finds it
    assert _call(mcp, "search", {"query": "doomed", "scope": "all"}) == []


def test_mcp_create_project_rejects_a_bad_slug(tmp_path):
    from mcp.server.fastmcp.exceptions import ToolError

    mcp = build_mcp(_container(tmp_path))
    with pytest.raises(ToolError):
        _call(mcp, "create_project", {"name": "Bad Slug"})  # spaces + uppercase


def test_mcp_update_and_list_projects(tmp_path):
    mcp = build_mcp(_container(tmp_path))
    _call(mcp, "create_project", {"name": "api"})
    _call(mcp, "create_project", {"name": "svc"})

    updated = json.loads(_call(mcp, "update_project", {"name": "api", "description": "the API"})[0])
    assert updated["description"] == "the API"

    listed = {json.loads(p)["slug"] for p in _call(mcp, "list_projects", {})}
    assert listed == {"api", "svc"}  # __global__ excluded


def test_mcp_remember_rejects_over_window_content(tmp_path):
    """An over-window memory surfaces an explicit, actionable tool error — not a
    silent truncation — so the calling agent can split and retry."""
    from mcp.server.fastmcp.exceptions import ToolError

    from mnemo.adapters.embedding.hash_embedder import HashEmbedder
    from mnemo.adapters.embedding.sync_embedding_scheduler import SyncEmbeddingScheduler
    from mnemo.adapters.session.in_process_session_provider import InProcessSessionProvider
    from mnemo.application.project_gate import ProjectGate
    from mnemo.application.use_cases.remember_memory import RememberMemoryUseCaseImpl

    container = _container(tmp_path)
    container.create_project.execute("api")
    embedder = HashEmbedder(max_input=3)
    container.remember = RememberMemoryUseCaseImpl(
        container.repository,
        SyncEmbeddingScheduler(embedder, container.repository),
        embedder,
        InProcessSessionProvider(),
        ProjectGate(container.projects),
    )
    mcp = build_mcp(container)

    with pytest.raises(ToolError) as exc:
        _call(mcp, "remember", {"content": "one two three four five", "project": "api"})
    message = str(exc.value)
    assert "limit" in message and "split" in message  # actionable
    assert container.repository.list_all() == []  # nothing stored on reject


def test_mcp_remember_rejects_a_re_keyed_duplicate(tmp_path):
    """Re-storing identical content under a new topic_key surfaces an explicit tool error
    instead of being silently dropped as a duplicate (the evolution never happening)."""
    from mcp.server.fastmcp.exceptions import ToolError

    container = _container(tmp_path)
    container.create_project.execute("api")
    mcp = build_mcp(container)

    _call(mcp, "remember", {"content": "auth uses jwt", "project": "api"})
    with pytest.raises(ToolError) as exc:
        _call(mcp, "remember", {
            "content": "auth uses jwt", "project": "api", "topic_key": "auth/model",
        })
    assert "topic_key" in str(exc.value)  # actionable: names the conflicting topic_key
    assert len(container.repository.list_all()) == 1  # only the original remains


def test_mcp_search_requires_a_project_in_project_scope(tmp_path):
    """A project-scoped search with no project surfaces an explicit, actionable
    tool error — not a silent empty result — so the calling agent can fix the call."""
    from mcp.server.fastmcp.exceptions import ToolError

    mcp = build_mcp(_container(tmp_path))
    with pytest.raises(ToolError) as exc:
        _call(mcp, "search", {"query": "anything"})  # scope defaults to 'project'
    message = str(exc.value)
    assert "scope='project'" in message and "project" in message  # actionable


def test_mcp_search_rejects_project_with_all_or_global_scope(tmp_path):
    """scope='all'/'global' ignore project; passing one is a loud error, not a
    silently wrong-scoped result."""
    from mcp.server.fastmcp.exceptions import ToolError

    mcp = build_mcp(_container(tmp_path))
    for scope in ("all", "global"):
        with pytest.raises(ToolError) as exc:
            _call(mcp, "search", {"query": "x", "scope": scope, "project": "api"})
        assert f"scope='{scope}'" in str(exc.value)


def test_mcp_browse_lists_memories_without_a_query(tmp_path):
    """The browse tool is callable via call_tool and returns recency-ordered hits
    that carry no score (no relevance ranking)."""
    mcp = build_mcp(_container(tmp_path))
    _call(mcp, "create_project", {"name": "api"})
    a = json.loads(_call(mcp, "remember", {"content": "alpha", "type": "decision", "project": "api"})[0])
    b = json.loads(_call(mcp, "remember", {"content": "beta", "project": "api"})[0])

    hits = [json.loads(block) for block in _call(mcp, "browse", {"project": "api"})]
    created = [hit["created_at"] for hit in hits]
    assert created == sorted(created, reverse=True)  # newest first
    assert {hit["id"] for hit in hits} == {a["id"], b["id"]}
    assert all("score" not in hit for hit in hits)  # browse carries no score


def test_mcp_browse_requires_a_project_in_project_scope(tmp_path):
    """Browse inherits the same scope/project guard as search."""
    from mcp.server.fastmcp.exceptions import ToolError

    mcp = build_mcp(_container(tmp_path))
    with pytest.raises(ToolError) as exc:
        _call(mcp, "browse", {})  # scope defaults to 'project', no project
    assert "scope='project'" in str(exc.value)


def test_mcp_remember_requires_a_project_in_project_scope(tmp_path):
    """The write path enforces the same contract as the read path: a project-scoped
    remember with no project is a loud error, not a silently unreachable row."""
    from mcp.server.fastmcp.exceptions import ToolError

    mcp = build_mcp(_container(tmp_path))
    with pytest.raises(ToolError) as exc:
        _call(mcp, "remember", {"content": "orphan note"})  # scope defaults to 'project'
    assert "scope='project'" in str(exc.value)


def test_mcp_remember_rejects_project_with_global_scope(tmp_path):
    from mcp.server.fastmcp.exceptions import ToolError

    mcp = build_mcp(_container(tmp_path))
    with pytest.raises(ToolError) as exc:
        _call(mcp, "remember", {"content": "a rule", "scope": "global", "project": "api"})
    assert "scope='global'" in str(exc.value)


def test_mcp_purge(tmp_path):
    mcp = build_mcp(_container(tmp_path))
    _call(mcp, "create_project", {"name": "api"})
    _call(mcp, "create_project", {"name": "other"})
    _call(mcp, "remember", {"content": "alpha", "project": "api"})
    _call(mcp, "remember", {"content": "beta", "project": "other"})

    assert json.loads(_call(mcp, "purge", {})[0])["deleted"] == 2
