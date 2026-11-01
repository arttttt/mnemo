import asyncio
import json

import pytest

pytest.importorskip("mcp")
pytest.importorskip("sqlite_vec")

from mnemo.adapters.mcp.server import build_mcp
from mnemo.application.fusion.fuser import Fuser
from mnemo.application.use_cases.recall_project import RecallProjectUseCaseImpl
from mnemo.domain.memory_type import MemoryType
from mnemo.infrastructure.composition import build_container
from mnemo.infrastructure.config import Config


def _container(tmp_path):
    config = Config(
        data_dir=str(tmp_path),
        embedder="hash",
        reranker="off",  # keep the MCP wiring test offline: the default reranker is a GGUF download
        sqlite_path=str(tmp_path / "memory.db"),
    )
    return build_container(config)


def _tools(tmp_path):
    mcp = build_mcp(_container(tmp_path))
    return {tool.name: tool for tool in asyncio.run(mcp.list_tools())}


def test_mcp_exposes_the_agent_tools(tmp_path):
    tools = set(_tools(tmp_path))
    assert {
        "remember", "search", "browse", "get", "recall", "delete",
        "create_project", "delete_project", "update_project", "list_projects",
    } <= tools
    assert "purge" not in tools  # drop-everything is too destructive for the agent surface (CLI-only)


def test_remember_advertises_allowed_types(tmp_path):
    schema = _tools(tmp_path)["remember"].input_schema
    type_schema = schema["properties"]["type"]
    enum = type_schema.get("enum")
    assert enum is not None, type_schema
    assert set(enum) == {member.value for member in MemoryType}


def test_optional_params_expose_concrete_types(tmp_path):
    # Optional[...] would render as anyOf[T, null] (some MCP clients show that as
    # "unknown"); every parameter must expose a concrete type or enum instead.
    for name, tool in _tools(tmp_path).items():
        for pname, pschema in (tool.input_schema.get("properties") or {}).items():
            assert "anyOf" not in pschema, f"{name}.{pname} uses anyOf"
            assert "type" in pschema or "enum" in pschema, f"{name}.{pname} lacks a type"


def test_only_required_params_are_marked_required(tmp_path):
    required = {
        name: tool.input_schema.get("required", [])
        for name, tool in _tools(tmp_path).items()
    }
    assert required["remember"] == ["content"]
    assert required["search"] == ["query"]
    assert required["browse"] == []  # query-less: every param is optional
    assert required["recall"] == ["query", "project"]
    assert required["delete"] == ["ids"]
    assert required["create_project"] == ["name"]
    assert required["delete_project"] == ["name"]
    assert required["update_project"] == ["name", "description"]
    assert required["list_projects"] == []


def _call(mcp, name, args):
    return [block.text for block in asyncio.run(mcp.call_tool(name, args)).content]


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


def test_mcp_search_hit_exposes_topic_key(tmp_path):
    """A search hit carries topic_key so the agent can pivot to get / remember by key."""
    mcp = build_mcp(_container(tmp_path))
    _call(mcp, "create_project", {"name": "api"})
    _call(mcp, "remember", {"content": "jwt refresh rotation", "type": "decision",
                            "project": "api", "topic_key": "auth/jwt"})
    hit = json.loads(_call(mcp, "search", {"query": "jwt rotation", "project": "api"})[0])
    assert hit["topic_key"] == "auth/jwt"


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
        container.repository, container.embedder, Fuser(),
        reranker=None, generator=_StubGenerator(),
        rerank_top_k=20, generator_max_tokens=128,
    )
    mcp = build_mcp(container)
    _call(mcp, "create_project", {"name": "api"})
    remembered = json.loads(
        _call(mcp, "remember", {"content": "jwt refresh rotation", "type": "decision", "project": "api"})[0]
    )

    # force=True acknowledges the experimental LLM-synthesis gate (see the gate test below)
    result = json.loads(_call(mcp, "recall", {"query": "auth", "project": "api", "force": True})[0])

    assert result["project"] == "api"
    assert result["summary"] == "auth uses jwt refresh rotation"  # the synthesized answer
    # the supporting memory comes back only as a light reference — id + type, never its content
    assert result["sources"] == [{"id": remembered["id"], "type": "decision"}]
    assert "sections" not in result  # full memory content is not dumped to the caller


def test_mcp_recall_refuses_without_force(tmp_path):
    """recall is experimental (LLM-synthesized, may be wrong); without force=true it
    refuses with an actionable warning instead of silently invoking the model. The gate
    fires before the use case runs, so no generator/model is touched."""
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = build_mcp(_container(tmp_path))  # default (heavy) generator — must never be reached
    _call(mcp, "create_project", {"name": "api"})

    with pytest.raises(ToolError) as exc:
        _call(mcp, "recall", {"query": "auth", "project": "api"})  # force defaults to False
    message = str(exc.value)
    assert "experimental" in message.lower()  # warns it may be wrong
    assert "force=true" in message  # actionable: tells the caller how to proceed


def test_mcp_recall_force_is_optional_and_concrete(tmp_path):
    """force carries a default (not required) and exposes a concrete boolean type so the
    tool schema stays clean for MCP clients."""
    tool = _tools(tmp_path)["recall"]
    assert tool.input_schema.get("required", []) == ["query", "project"]  # force not required
    assert tool.input_schema["properties"]["force"]["type"] == "boolean"


def test_mcp_delete_cascade_removes_the_whole_supersede_lineage(tmp_path):
    """cascade=true deletes a memory plus every older version it supersedes — the whole
    chain. An agent cannot do this by id: search/browse surface only the active head, so the
    superseded versions are invisible and unenumerable."""
    mcp = build_mcp(_container(tmp_path))
    _call(mcp, "create_project", {"name": "api"})
    _call(mcp, "remember", {"content": "auth v1", "project": "api", "topic_key": "auth/model"})
    _call(mcp, "remember", {"content": "auth v2", "project": "api", "topic_key": "auth/model"})
    head = json.loads(
        _call(mcp, "remember", {"content": "auth v3", "project": "api", "topic_key": "auth/model"})[0]
    )

    assert len(_call(mcp, "search", {"query": "auth", "project": "api"})) == 1  # only the head is visible
    deleted = json.loads(_call(mcp, "delete", {"ids": [head["id"]], "cascade": True})[0])
    assert deleted["deleted"] == 3  # head + the two superseded versions
    assert _call(mcp, "search", {"query": "auth", "project": "api"}) == []  # the lineage is gone


def test_mcp_delete_project_cascades(tmp_path):
    mcp = build_mcp(_container(tmp_path))
    _call(mcp, "create_project", {"name": "api"})
    _call(mcp, "remember", {"content": "doomed via mcp", "project": "api"})

    deleted = json.loads(_call(mcp, "delete_project", {"name": "api"})[0])
    assert deleted["slug"] == "api"
    # its memory cascaded away — a cross-project search no longer finds it
    assert _call(mcp, "search", {"query": "doomed", "scope": "all"}) == []


def test_mcp_create_project_rejects_a_bad_slug(tmp_path):
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = build_mcp(_container(tmp_path))
    with pytest.raises(ToolError, match=r"invalid project slug 'Bad Slug': use kebab-case"):
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
    from mcp.server.mcpserver.exceptions import ToolError

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
    from mcp.server.mcpserver.exceptions import ToolError

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
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = build_mcp(_container(tmp_path))
    with pytest.raises(ToolError) as exc:
        _call(mcp, "search", {"query": "anything"})  # scope defaults to 'project'
    message = str(exc.value)
    assert "scope='project'" in message and "project" in message  # actionable


def test_mcp_search_rejects_project_with_all_or_global_scope(tmp_path):
    """scope='all'/'global' ignore project; passing one is a loud error, not a
    silently wrong-scoped result."""
    from mcp.server.mcpserver.exceptions import ToolError

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
    assert all("topic_key" in hit and hit["status"] == "active" for hit in hits)  # audit fields on hits


def test_mcp_browse_requires_a_project_in_project_scope(tmp_path):
    """Browse inherits the same scope/project guard as search."""
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = build_mcp(_container(tmp_path))
    with pytest.raises(ToolError) as exc:
        _call(mcp, "browse", {})  # scope defaults to 'project', no project
    assert "scope='project'" in str(exc.value)


def test_mcp_get_dereferences_by_topic_key_with_its_chain(tmp_path):
    """get(topic_key=...) returns the active head + its supersede chain via call_tool."""
    mcp = build_mcp(_container(tmp_path))
    _call(mcp, "create_project", {"name": "api"})
    _call(mcp, "remember", {"content": "auth v1", "project": "api", "topic_key": "auth/model"})
    head = json.loads(_call(mcp, "remember", {"content": "auth v2", "project": "api", "topic_key": "auth/model"})[0])

    result = json.loads(_call(mcp, "get", {"topic_key": "auth/model", "project": "api"})[0])
    assert result["id"] == head["id"] and result["content"] == "auth v2"
    assert result["status"] == "active" and result["topic_key"] == "auth/model"
    assert [e["status"] for e in result["chain"]] == ["active", "superseded"]
    assert result["chain_total"] == 2


def test_mcp_get_by_id_reaches_a_superseded_version(tmp_path):
    """get(id=...) reaches a record search/browse cannot surface (a superseded version)."""
    mcp = build_mcp(_container(tmp_path))
    _call(mcp, "create_project", {"name": "api"})
    old = json.loads(_call(mcp, "remember", {"content": "auth v1", "project": "api", "topic_key": "auth/model"})[0])
    _call(mcp, "remember", {"content": "auth v2", "project": "api", "topic_key": "auth/model"})

    result = json.loads(_call(mcp, "get", {"id": old["id"]})[0])
    assert result["content"] == "auth v1" and result["status"] == "superseded"


def test_mcp_get_missing_topic_key_is_a_loud_error(tmp_path):
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = build_mcp(_container(tmp_path))
    _call(mcp, "create_project", {"name": "api"})
    with pytest.raises(ToolError) as exc:
        _call(mcp, "get", {"topic_key": "nope/x", "project": "api"})
    assert "topic_key" in str(exc.value)


def test_mcp_remember_requires_a_project_in_project_scope(tmp_path):
    """The write path enforces the same contract as the read path: a project-scoped
    remember with no project is a loud error, not a silently unreachable row."""
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = build_mcp(_container(tmp_path))
    with pytest.raises(ToolError) as exc:
        _call(mcp, "remember", {"content": "orphan note"})  # scope defaults to 'project'
    assert "scope='project'" in str(exc.value)


def test_mcp_remember_rejects_project_with_global_scope(tmp_path):
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = build_mcp(_container(tmp_path))
    with pytest.raises(ToolError) as exc:
        _call(mcp, "remember", {"content": "a rule", "scope": "global", "project": "api"})
    assert "scope='global'" in str(exc.value)


# What each tool does to the store: the side-effect class it declares to clients. Clients
# act on it — Codex runs a read-only tool without asking and asks before a destructive one;
# Claude Code runs read-only calls in parallel. Listed by hand, so a new tool cannot reach
# agents without someone deciding its class here.
_SIDE_EFFECTS = {
    "search": "reads",
    "browse": "reads",
    "get": "reads",
    "recall": "reads",
    "list_projects": "reads",
    "remember": "writes",
    "create_project": "writes",
    "update_project": "writes",
    "delete": "deletes",
    "delete_project": "deletes",
}
_DECLARED_HINTS = {
    "reads": {"readOnlyHint": True, "openWorldHint": False},
    "writes": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    "deletes": {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": False},
}
# One call per reading tool, against the store _remembered_twice builds.
_READS = {
    "search": {"query": "jwt rotation", "project": "api"},
    "browse": {"project": "api"},
    "get": {"topic_key": "auth/jwt", "project": "api"},
    "recall": {"query": "auth", "project": "api", "force": True},
    "list_projects": {},
}


def test_every_tool_declares_its_side_effect_class(tmp_path):
    declared = {
        name: tool.annotations.model_dump(by_alias=True, exclude_none=True)
        if tool.annotations else None
        for name, tool in _tools(tmp_path).items()
    }
    assert declared == {name: _DECLARED_HINTS[kind] for name, kind in _SIDE_EFFECTS.items()}


def _store_rows(db_path) -> dict:
    """Every row of every table — FTS and vector tables included — to compare a store
    before and after a call."""
    import sqlite3

    import sqlite_vec

    connection = sqlite3.connect(db_path)
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        tables = [name for (name,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )]
        return {table: sorted(map(repr, connection.execute(f'SELECT * FROM "{table}"')))
                for table in tables}
    finally:
        connection.close()


class _QuietGenerator:
    """Stands in for the recall model, which is too heavy for a test; recall still reads
    the store end to end."""

    def generate(self, prompt, *, max_tokens):
        return "an answer"


def _remembered_twice(tmp_path):
    """A server over a store holding one project and a two-version topic_key chain."""
    container = _container(tmp_path)
    container.recall = RecallProjectUseCaseImpl(
        container.repository, container.embedder, Fuser(),
        reranker=None, generator=_QuietGenerator(),
        rerank_top_k=20, generator_max_tokens=128,
    )
    mcp = build_mcp(container)
    _call(mcp, "create_project", {"name": "api"})
    for content in ("jwt refresh rotation", "jwt refresh rotation, every 15 minutes"):
        _call(mcp, "remember", {"content": content, "type": "decision",
                                "project": "api", "topic_key": "auth/jwt"})
    return mcp


def test_the_read_calls_cover_every_reading_tool():
    assert set(_READS) == {name for name, kind in _SIDE_EFFECTS.items() if kind == "reads"}


@pytest.mark.parametrize("name", sorted(_READS))
def test_a_reading_tool_leaves_the_store_unchanged(tmp_path, name):
    """A tool declared read-only is run without the user's approval and alongside other
    calls, so it must not change the store at all."""
    mcp = _remembered_twice(tmp_path)
    before = _store_rows(tmp_path / "memory.db")

    assert _call(mcp, name, _READS[name])  # the call did read something

    assert _store_rows(tmp_path / "memory.db") == before
