"""The boundary mnemo puts around every MCP tool call (MnemoMCPServer.call_tool)."""
import logging
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

import anyio
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError

from mnemo.adapters.mcp.tool_boundary import MnemoMCPServer
from mnemo.adapters.session.meta_session_provider import SESSION_META_KEY, MetaSessionProvider

_NO_SESSION = "<no session>"


class _Occupancy:
    """How many tool bodies run at once (tools run on worker threads)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def __enter__(self):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def __exit__(self, *exc):
        with self._lock:
            self.active -= 1


def _server(occupancy: _Occupancy | None = None) -> MnemoMCPServer:
    server = MnemoMCPServer("boundary-test")
    provider = MetaSessionProvider()
    occupancy = occupancy or _Occupancy()

    @server.tool()
    def whoami() -> str:  # sync, so the SDK runs it on a worker thread
        return provider.current_session_id() or _NO_SESSION

    @server.tool()
    def crash() -> str:
        raise RuntimeError("disk on fire")

    @server.tool()
    def slow() -> str:
        with occupancy:
            time.sleep(0.1)
        return "done"

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


def test_concurrent_calls_run_one_at_a_time():
    """Calls from several agents arrive concurrently; their tool bodies must never overlap."""
    occupancy = _Occupancy()
    server = _server(occupancy)

    async def three_at_once():
        with anyio.fail_after(5):  # a call slot that is never released fails, not hangs
            async with anyio.create_task_group() as group:
                for _ in range(3):
                    group.start_soon(server.call_tool, "slow", {}, None)

    anyio.run(three_at_once)
    assert occupancy.peak == 1


def test_a_failed_call_does_not_block_the_next():
    server = _server()

    async def crash_then_call():
        with pytest.raises(ToolError):
            await server.call_tool("crash", {}, None)
        with anyio.fail_after(2):  # would hang if the failure kept the call slot
            return await server.call_tool("whoami", {}, None)

    assert anyio.run(crash_then_call).content[0].text == _NO_SESSION


def test_a_call_abandoned_while_waiting_does_not_block_the_next():
    server = _server()

    async def abandon_one_then_call():
        async with anyio.create_task_group() as group:
            group.start_soon(server.call_tool, "slow", {}, None)
            await anyio.sleep(0.02)  # the slow call is running
            with anyio.move_on_after(0.02):  # this caller gives up while still waiting
                await server.call_tool("whoami", {}, None)
        with anyio.fail_after(2):  # would hang if the abandoned wait kept the call slot
            return await server.call_tool("whoami", {}, None)

    assert anyio.run(abandon_one_then_call).content[0].text == _NO_SESSION
