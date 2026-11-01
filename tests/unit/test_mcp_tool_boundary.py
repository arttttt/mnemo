"""The boundary mnemo puts around every MCP tool call (MnemoMCPServer.call_tool)."""
import logging
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

import anyio
import mcp.types as types
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError

from mnemo.adapters.mcp.tool_boundary import MnemoMCPServer, _structured_only
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


def _server(occupancy: _Occupancy | None = None, entered: list | None = None) -> MnemoMCPServer:
    server = MnemoMCPServer("boundary-test")
    provider = MetaSessionProvider()
    occupancy = occupancy or _Occupancy()
    entered = entered if entered is not None else []

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

    @server.tool()
    def record(tag: int) -> str:
        entered.append(tag)
        return str(tag)

    return server


def _run_guarded(scenario, timeout: float = 10.0):
    """Run an async ``scenario`` on its own thread under a watchdog. A call slot that blocks
    the event loop itself cannot be interrupted by any deadline inside that loop; this
    turns such a hang into a failed test instead of a stuck suite."""
    outcome = {}

    def target():
        try:
            outcome["value"] = anyio.run(scenario)
        except BaseException as exc:  # handed back to the test thread below
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        pytest.fail(f"the event loop was still blocked after {timeout}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def _request(server, meta) -> Context:
    """A Context for a call that arrived as a request carrying ``meta`` as its `_meta`."""
    return Context(request_context=SimpleNamespace(meta=meta), mcp_server=server)


def _answer(result):
    """A str-returning tool's answer: structured {"result": ...} only, no text copy."""
    assert result.content == []
    return result.structured_content["result"]


def test_the_session_sent_with_the_request_is_bound_for_the_call_only():
    server = _server()

    async def call_then_read_after():
        result = await server.call_tool("whoami", {}, _request(server, {SESSION_META_KEY: "s-1"}))
        return _answer(result), MetaSessionProvider().current_session_id()

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
    assert _answer(result) == _NO_SESSION


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
        async with anyio.create_task_group() as group:
            for _ in range(3):
                group.start_soon(server.call_tool, "slow", {}, None)

    _run_guarded(three_at_once)
    assert occupancy.peak == 1


def test_waiting_calls_run_in_arrival_order():
    entered = []
    server = _server(entered=entered)

    async def queue_three_behind_a_slow_call():
        async with anyio.create_task_group() as group:
            group.start_soon(server.call_tool, "slow", {}, None)
            await anyio.sleep(0.02)  # the slow call holds the slot
            for tag in (1, 2, 3):
                group.start_soon(server.call_tool, "record", {"tag": tag}, None)
                await anyio.sleep(0.005)  # arrive strictly in this order

    _run_guarded(queue_three_behind_a_slow_call)
    assert entered == [1, 2, 3]


def test_the_server_stays_responsive_while_calls_wait():
    """One call runs, another waits for its turn — and requests that need no call slot (here,
    listing the tools) are still answered meanwhile: the waiting does not block the loop."""
    occupancy = _Occupancy()
    server = _server(occupancy)

    async def list_tools_while_a_call_waits():
        async with anyio.create_task_group() as group:
            group.start_soon(server.call_tool, "slow", {}, None)
            await anyio.sleep(0.02)  # the slow call holds the slot
            group.start_soon(server.call_tool, "whoami", {}, None)  # waits for it
            await anyio.sleep(0.01)
            tools = await server.list_tools()
            still_running = occupancy.active
        return len(tools), still_running

    listed, still_running = _run_guarded(list_tools_while_a_call_waits)
    assert listed == 4 and still_running == 1  # answered before the slow call finished


def test_a_failed_call_does_not_block_the_next():
    server = _server()

    async def crash_then_call():
        with pytest.raises(ToolError):
            await server.call_tool("crash", {}, None)
        return await server.call_tool("whoami", {}, None)

    assert _answer(_run_guarded(crash_then_call)) == _NO_SESSION


def test_a_call_abandoned_while_waiting_does_not_block_the_next():
    server = _server()

    async def abandon_one_then_call():
        async with anyio.create_task_group() as group:
            group.start_soon(server.call_tool, "slow", {}, None)
            await anyio.sleep(0.02)  # the slow call is running
            with anyio.move_on_after(0.02):  # this caller gives up while still waiting
                await server.call_tool("whoami", {}, None)
        return await server.call_tool("whoami", {}, None)

    assert _answer(_run_guarded(abandon_one_then_call)) == _NO_SESSION


def test_only_a_successful_structured_answer_loses_its_text_copy():
    text = [types.TextContent(type="text", text='{"result": 1}')]
    structured = types.CallToolResult(content=text, structured_content={"result": 1})
    error = types.CallToolResult(content=text, structured_content={"result": 1}, is_error=True)
    unstructured = types.CallToolResult(content=text)

    assert _structured_only(structured).content == []
    assert _structured_only(structured).structured_content == {"result": 1}
    assert _structured_only(error).content == text  # an error is read from its text
    assert _structured_only(unstructured).content == text  # nothing else to read
