"""Client-side plumbing for tests that drive mnemo over real MCP transports.

Assertions compare :func:`wire` — the JSON an MCP client receives (camelCase keys, unset
fields omitted) — so the same expectations hold whichever MCP SDK version is underneath.
Only the connection helpers here touch the SDK's client API.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"


def service_env(host: str, port: int, data_dir) -> dict:
    """Env for a `mnemo-service` / `mnemo-mcp` subprocess (offline: SQLite + hash embedder)."""
    return {
        **os.environ,
        "PYTHONPATH": str(SRC),
        "MNEMO_EMBEDDER": "hash",
        "MNEMO_DATA_DIR": str(data_dir),
        "MNEMO_HOST": host,
        "MNEMO_PORT": str(port),
    }


def wire(model) -> dict:
    """The JSON an MCP client receives for a protocol result.

    The SDK's client model fills in ``resultType: "complete"`` for a result that arrives
    without one — and on the protocol versions these sessions negotiate the server never
    sends it (it is sieved out for pre-2026 versions) — so that default is dropped: it
    was not on the wire."""
    dumped = model.model_dump(mode="json", by_alias=True, exclude_none=True)
    if dumped.get("resultType") == "complete":
        del dumped["resultType"]
    return dumped


async def over_http(url: str, act):
    """Run ``act(session)`` in one streamable-http client session against the service."""
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await act(session)


async def through_proxy(host: str, port: int, data_dir, act):
    """Run ``act(session)`` in one stdio client session through a `mnemo-mcp` proxy process
    pointed at ``host:port``."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from mnemo.adapters.mcp.proxy import main; main()"],
        env=service_env(host, port, data_dir),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await act(session)
