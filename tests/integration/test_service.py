"""The shared service serves MCP over streamable-http, and several agents share it.

Drives the real boundary: a streamable-http MCP client against a running
`mnemo-service` subprocess (offline: memory store + hash embedder).
"""
import pytest

pytest.importorskip("mcp")
pytest.importorskip("uvicorn")


async def _call(url: str, tool: str, args: dict):
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool, args)


def _text(result) -> str:
    return " ".join(getattr(block, "text", "") or "" for block in result.content)


def test_service_serves_tools_over_http(service):
    import anyio

    host, port = service
    url = f"http://{host}:{port}/mcp"

    stored = anyio.run(_call, url, "remember", {"content": "redis caching layer", "project": "api"})
    assert not stored.isError, _text(stored)

    hits = anyio.run(_call, url, "search", {"query": "redis cache", "scope": "all"})
    assert not hits.isError
    assert "redis caching layer" in _text(hits)


def test_several_connections_share_one_service(service):
    import anyio

    host, port = service
    url = f"http://{host}:{port}/mcp"

    # One connection writes; a SEPARATE connection reads it back → both are served
    # by the one process (the single shared embedder + store behind them).
    anyio.run(_call, url, "remember", {"content": "jwt rotation policy", "project": "api"})
    hits = anyio.run(_call, url, "search", {"query": "jwt rotation", "scope": "all"})
    assert "jwt rotation policy" in _text(hits)
