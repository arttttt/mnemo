"""The thin per-agent connector: a stdio MCP proxy into the shared service.

This is what each agent's MCP client launches (`mnemo-mcp`). It loads nothing
heavy — it forwards initialize / list_tools / call_tool to the shared
`mnemo-service` over streamable-http, so the embedder and store live once in
that one process regardless of how many agents are connected.
"""
from __future__ import annotations

from mnemo.infrastructure.config import Config


async def _serve(url: str) -> None:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    async with streamable_http_client(url) as (http_read, http_write, _):
        async with ClientSession(http_read, http_write) as upstream:
            await upstream.initialize()
            server = Server("mnemo")

            @server.list_tools()
            async def list_tools():
                return (await upstream.list_tools()).tools

            @server.call_tool(validate_input=False)
            async def call_tool(name, arguments):
                # Forward the upstream result verbatim (content + structured + isError).
                return await upstream.call_tool(name, arguments)

            async with stdio_server() as (stdio_read, stdio_write):
                await server.run(
                    stdio_read, stdio_write, server.create_initialization_options()
                )


def main() -> None:
    import anyio

    config = Config.from_env()
    anyio.run(_serve, f"http://{config.host}:{config.port}/mcp")
