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


def test_proxy_forwards_agent_calls_to_the_service(service, tmp_path):
    import anyio

    host, port = service

    async def flow():
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=["-c", "from mnemo.adapters.mcp.proxy import main; main()"],
            env={
                **os.environ,
                "PYTHONPATH": str(_SRC),
                "MNEMO_DATA_DIR": str(tmp_path),
                "MNEMO_HOST": host,
                "MNEMO_PORT": str(port),
            },
        )
        async with stdio_client(params) as (read, write):
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
