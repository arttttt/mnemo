"""Client-side plumbing for tests that drive mnemo over real MCP transports.

Assertions compare :func:`wire` — the JSON an MCP client receives (camelCase keys, unset
fields omitted) — so the same expectations hold whichever MCP SDK version is underneath.
Only the connection helpers here touch the SDK's client API.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"


def answer_text(result) -> str:
    """What the agent reads from a tool result: its structured answer as JSON, or the text
    of an error (mnemo sends no text copy of a structured answer)."""
    if result.structured_content is not None:
        return json.dumps(result.structured_content, ensure_ascii=False)
    return " ".join(getattr(block, "text", "") or "" for block in result.content)


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
    """The JSON an MCP client received for a protocol result: exactly the fields the server
    sent. Fields the client model fills in with a default when they are absent (such as
    the 2.x ``resultType``) are not part of it."""
    return model.model_dump(mode="json", by_alias=True, exclude_unset=True)


async def over_http(url: str, act):
    """Run ``act(session)`` in one streamable-http client session against the service."""
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await act(session)


def proxy_params(host: str, port: int, data_dir):
    """How a stdio MCP client launches a `mnemo-mcp` proxy pointed at ``host:port``."""
    from mcp.client.stdio import StdioServerParameters

    return StdioServerParameters(
        command=sys.executable,
        args=["-c", "from mnemo.adapters.mcp.proxy import main; main()"],
        env=service_env(host, port, data_dir),
    )


async def through_proxy(host: str, port: int, data_dir, act):
    """Run ``act(session)`` in one stdio client session through a `mnemo-mcp` proxy process
    pointed at ``host:port``."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(proxy_params(host, port, data_dir)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await act(session)
