"""The boundary mnemo puts around every MCP tool call (MnemoMCPServer.call_tool)."""
import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

import anyio
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError

from mnemo.adapters.mcp.tool_boundary import MnemoMCPServer
from mnemo.adapters.session.meta_session_provider import SESSION_META_KEY, MetaSessionProvider

_NO_SESSION = "<no session>"


def _server() -> MnemoMCPServer:
    server = MnemoMCPServer("boundary-test")
    provider = MetaSessionProvider()

    @server.tool()
    def whoami() -> str:  # sync, so the SDK runs it on a worker thread
        return provider.current_session_id() or _NO_SESSION

    @server.tool()
    def crash() -> str:
        raise RuntimeError("disk on fire")

    return server


def _request(server, meta) -> Context:
    """A Context for a call that arrived as a request carrying ``meta`` as its `_meta`."""
    return Context(request_context=SimpleNamespace(meta=meta), mcp_server=server)


def test_the_session_sent_with_the_request_is_bound_for_the_call_only():
    server = _server()

    async def call_then_read_after():
        result = await server.call_tool("whoami", {}, _request(server, {SESSION_META_KEY: "s-1"}))
        return result.content[0].text, MetaSessionProvider().current_session_id()

    during, after = anyio.run(call_then_read_after)
    assert during == "s-1"  # visible to the tool on its worker thread
    assert after is None  # unbound once the call returns


@pytest.mark.parametrize(
    "context",
    [
        lambda server: None,  # an in-process call with no Context at all
        lambda server: Context(mcp_server=server),  # a Context built outside a request
        lambda server: _request(server, None),  # a request without `_meta`
        lambda server: _request(server, {"other": "x"}),  # `_meta` without our key
    ],
    ids=["no-context", "outside-request", "no-meta", "meta-without-session"],
)
def test_a_call_without_a_session_id_runs_without_one(context):
    server = _server()
    result = anyio.run(server.call_tool, "whoami", {}, context(server))
    assert result.content[0].text == _NO_SESSION


def test_a_crash_reaches_the_caller_with_its_message_and_is_logged(caplog):
    server = _server()

    with caplog.at_level(logging.INFO, logger="mnemo.mcp"):
        with pytest.raises(ToolError, match=r"^Error executing tool crash: disk on fire$"):
            anyio.run(server.call_tool, "crash", {}, None)

    (record,) = [r for r in caplog.records if r.name == "mnemo.mcp"]
    assert record.levelno == logging.INFO
    assert isinstance(record.exc_info[1], RuntimeError)  # the traceback of the real cause
