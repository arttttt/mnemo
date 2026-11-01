"""What an agent sees on the MCP wire — pinned, so an SDK upgrade cannot change it silently.

Every assertion is on the JSON a client receives (see ``tests.support.mcp_wire.wire``),
over the real boundaries: a streamable-http client against `mnemo-service`, and a stdio
client through the `mnemo-mcp` proxy in front of it.
"""
import json
import os
import socket
import threading
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("uvicorn")
pytest.importorskip("sqlite_vec")

from tests.support.mcp_wire import over_http, through_proxy, wire

_GOLDEN_TOOLS = Path(__file__).parent / "golden" / "mcp_tools.json"
_DEFINITION_KEYS = ("name", "description", "inputSchema", "outputSchema")

# Failing calls whose message mnemo writes. The agent reads it after the prefix
# "Error executing tool <name>: "; the envelope and the prefix are pinned exactly, the
# message only by a phrase that identifies it — its wording belongs to the code that
# raises it and is checked there. (The near-match case needs a registered "api" project.)
_TOOL_FAILURES = [
    ("search", {"query": "x"}, "scope='project' needs a project to scope to"),
    ("search", {"query": "x", "project": "ap"}, "unknown project 'ap'"),
    ("recall", {"query": "x", "project": "api"}, "call recall again with force=true"),
    ("create_project", {"name": "Bad Slug"}, "invalid project slug 'Bad Slug'"),
]
# Failures whose message the SDK writes: pinned exactly.
_UNKNOWN_TOOL = ("nope_tool", {}, "Unknown tool: nope_tool")
# An argument the tool's schema rejects. pydantic writes the message, and its tail links
# to a docs URL that embeds the pydantic version, so only the stable head is pinned.
_BAD_ARGUMENT = ("search", {"query": "x", "scope": "nope", "project": "api"})
_BAD_ARGUMENT_HEAD = (
    "Error executing tool search: 1 validation error for searchArguments\nscope\n"
    "  Input should be 'project', 'global' or 'all' [type=literal_error, "
    "input_value='nope', input_type=str]"
)


def _url(host: str, port: int) -> str:
    return f"http://{host}:{port}/mcp"


def _run(fn, *args):
    import anyio

    return anyio.run(fn, *args)


def _definitions(listed) -> dict:
    return {
        tool["name"]: {key: tool[key] for key in _DEFINITION_KEYS if key in tool}
        for tool in wire(listed)["tools"]
    }


def _pretty(text: str) -> str:
    return json.dumps(json.loads(text), indent=2, ensure_ascii=False)


def test_tool_definitions_match_the_pinned_contract(service, tmp_path):
    """Names, descriptions and input/output schemas of every tool, as listed over HTTP and
    through the proxy. Regenerate deliberately with MNEMO_UPDATE_GOLDEN=1."""
    host, port = service
    served = _definitions(_run(over_http, _url(host, port), lambda s: s.list_tools()))
    relayed = _definitions(_run(through_proxy, host, port, tmp_path, lambda s: s.list_tools()))

    if os.environ.get("MNEMO_UPDATE_GOLDEN") == "1":
        _GOLDEN_TOOLS.write_text(json.dumps(served, indent=2, ensure_ascii=False) + "\n")
    assert served == json.loads(_GOLDEN_TOOLS.read_text())
    assert relayed == served  # the proxy lists exactly what the service serves


def _assert_dict_result(result: dict) -> dict:
    """A dict result: one text block of 2-space-indented JSON, no structured copy."""
    assert set(result) == {"content", "isError"} and result["isError"] is False
    (block,) = result["content"]
    assert block["type"] == "text" and block["text"] == _pretty(block["text"])
    return json.loads(block["text"])


def test_successful_results_keep_their_shape(service, tmp_path):
    """dict, list[dict] and empty-list results as the service serves them — and relayed
    unchanged by the proxy."""
    host, port = service

    async def write(session):
        created = await session.call_tool("create_project", {"name": "api"})
        await session.call_tool(
            "remember", {"content": "jwt refresh rotation", "project": "api", "topic_key": "auth/jwt"}
        )
        return wire(created)

    async def reads(session):
        calls = [
            ("get", {"topic_key": "auth/jwt", "project": "api"}),
            ("search", {"query": "jwt", "project": "api"}),
            ("search", {"query": "nothing here", "scope": "global"}),
        ]
        return [wire(await session.call_tool(name, args)) for name, args in calls]

    created = _run(over_http, _url(host, port), write)
    served = _run(over_http, _url(host, port), reads)
    relayed = _run(through_proxy, host, port, tmp_path, reads)
    record, hits, empty = served

    assert _assert_dict_result(created)["slug"] == "api"
    assert _assert_dict_result(record)["topic_key"] == "auth/jwt"

    # A list[dict] result: one text block per item, plus the items as {"result": [...]}.
    assert hits["isError"] is False
    items = hits["structuredContent"]["result"]
    assert [item["content"] for item in items] == ["jwt refresh rotation"]
    assert all(b["type"] == "text" and b["text"] == _pretty(b["text"]) for b in hits["content"])
    assert [json.loads(b["text"]) for b in hits["content"]] == items

    # An empty list: no blocks and an empty structured result.
    assert empty == {"content": [], "structuredContent": {"result": []}, "isError": False}

    assert relayed == served  # the proxy relays successful results unchanged


async def _server_identity(session) -> dict:
    return {"name": session.server_info.name, "version": session.server_info.version}


def test_the_handshake_names_mnemo_and_its_version(service, tmp_path):
    """Both the service and the proxy introduce themselves as mnemo at its installed version."""
    from mnemo.infrastructure.package_version import installed_version

    host, port = service
    expected = {"name": "mnemo", "version": installed_version()}
    assert _run(over_http, _url(host, port), _server_identity) == expected
    assert _run(through_proxy, host, port, tmp_path, _server_identity) == expected


def test_tool_failures_reach_the_agent_as_readable_errors(service, tmp_path):
    """A failing call is a tool result the agent can read — isError plus the exact message —
    whether it comes straight from the service or through the proxy."""
    host, port = service
    _run(over_http, _url(host, port), lambda s: s.call_tool("create_project", {"name": "api"}))

    async def failing_calls(session):
        calls = [(name, args) for name, args, _ in _TOOL_FAILURES]
        calls += [_UNKNOWN_TOOL[:2], _BAD_ARGUMENT]
        return [wire(await session.call_tool(name, args)) for name, args in calls]

    served = _run(over_http, _url(host, port), failing_calls)
    relayed = _run(through_proxy, host, port, tmp_path, failing_calls)

    for (name, args, phrase), result in zip(_TOOL_FAILURES, served):
        assert set(result) == {"content", "isError"} and result["isError"] is True, (name, args)
        (block,) = result["content"]
        prefix = f"Error executing tool {name}: "
        assert block["type"] == "text" and block["text"].startswith(prefix), (name, args)
        assert phrase in block["text"][len(prefix):], (name, args)
    unknown_tool = served[len(_TOOL_FAILURES)]
    assert unknown_tool == {"content": [{"type": "text", "text": _UNKNOWN_TOOL[2]}], "isError": True}
    bad_argument = served[-1]
    assert set(bad_argument) == {"content", "isError"} and bad_argument["isError"] is True
    (block,) = bad_argument["content"]
    assert block["type"] == "text" and block["text"].startswith(_BAD_ARGUMENT_HEAD)
    assert relayed == served  # the proxy relays every failure verbatim


def test_proxy_reports_a_failed_forward_as_a_tool_error(tmp_path):
    """The service port accepts and drops every connection: the proxy sees a live service
    (so it does not spawn one) but each forward fails. The agent must still get a tool
    result it can read, not a protocol-level error."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(0.1)
    stop = threading.Event()

    def drop_every_connection():
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            connection.close()

    dropper = threading.Thread(target=drop_every_connection, daemon=True)
    dropper.start()
    try:
        result = wire(_run(
            through_proxy, "127.0.0.1", listener.getsockname()[1], tmp_path / "data",
            lambda s: s.call_tool("search", {"query": "x", "scope": "all"}),
        ))
    finally:
        stop.set()
        dropper.join()
        listener.close()

    assert set(result) == {"content", "isError"} and result["isError"] is True
    (block,) = result["content"]
    assert block["type"] == "text" and block["text"]


def test_proxy_relays_a_result_larger_than_one_mebibyte(service, tmp_path):
    """browse(limit=100) over large memories is one response well over 1 MiB; the proxy must
    relay it whole rather than fail on the size."""
    from mnemo.adapters.embedding.hash_embedder import HashEmbedder
    from mnemo.domain.memory import Memory
    from tests.support.sqlite_store import open_store

    host, port = service
    embedder = HashEmbedder()
    repo, _ = open_store(tmp_path, embedder.dim, projects=("api",))  # the service's own store
    for i in range(100):
        content = f"note {i:03d} " + "x" * 12_000
        repo.add(Memory.create(content, project="api"), embedder.encode(content))

    result = wire(_run(
        through_proxy, host, port, tmp_path,
        lambda s: s.call_tool("browse", {"project": "api", "limit": 100}),
    ))

    assert result["isError"] is False, result["content"][:1]
    assert len(result["content"]) == 100
    assert len(json.dumps(result)) > 1_048_576  # the relayed result really exceeds 1 MiB
