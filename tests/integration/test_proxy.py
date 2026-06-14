"""The `mnemo-mcp` proxy forwards agent calls to the shared service.

Exercises BOTH components together along the full agent path: a stdio MCP client
-> the `mnemo-mcp` proxy (subprocess) -> the `mnemo-service` (subprocess) -> the
store, and back.
"""
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("uvicorn")

_SRC = Path(__file__).resolve().parents[2] / "src"


def _text(result) -> str:
    return " ".join(getattr(block, "text", "") or "" for block in result.content)


def _proxy_params(host: str, port: int, data_dir):
    from mcp.client.stdio import StdioServerParameters

    return StdioServerParameters(
        command=sys.executable,
        args=["-c", "from mnemo.adapters.mcp.proxy import main; main()"],
        env={
            **os.environ,
            "PYTHONPATH": str(_SRC),
            "MNEMO_DATA_DIR": str(data_dir),
            "MNEMO_HOST": host,
            "MNEMO_PORT": str(port),
        },
    )


def test_proxy_forwards_agent_calls_to_the_service(service, tmp_path):
    import anyio

    host, port = service

    async def flow():
        from mcp.client.session import ClientSession
        from mcp.client.stdio import stdio_client

        async with stdio_client(_proxy_params(host, port, tmp_path)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # The proxy exposes the service's tools (it knows nothing itself).
                tools = {tool.name for tool in (await session.list_tools()).tools}
                assert {"remember", "search"} <= tools

                stored = await session.call_tool(
                    "remember", {"content": "redis via the proxy", "project": "api"}
                )
                assert not stored.isError, _text(stored)

                hits = await session.call_tool("search", {"query": "redis", "scope": "all"})
                assert "redis via the proxy" in _text(hits)

    anyio.run(flow)


async def _proxy_writes(host: str, port: int, data_dir, contents: list[str]) -> None:
    """One proxy run (one agent) that writes several memories through it."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(_proxy_params(host, port, data_dir)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for content in contents:
                await session.call_tool("remember", {"content": content, "project": "api"})


def test_each_proxy_run_stamps_its_own_session_id(service, tmp_path):
    import json

    import anyio

    host, port = service

    # One proxy run (one agent) writes twice → those memories share its session id.
    anyio.run(_proxy_writes, host, port, tmp_path, ["a1 via proxy one", "a2 via proxy one"])
    # A second, separate proxy run → its own session id.
    anyio.run(_proxy_writes, host, port, tmp_path, ["b1 via proxy two"])

    rows = json.loads((tmp_path / "memory.json").read_text())["memories"]
    session_id = {row["memory"]["content"]: row["memory"]["session_id"] for row in rows}
    assert session_id["a1 via proxy one"] == session_id["a2 via proxy one"]  # one run = one id
    assert session_id["b1 via proxy two"] != session_id["a1 via proxy one"]  # other agent = other id
    assert all(session_id.values())  # the service stamped the proxy-supplied id (not null)
