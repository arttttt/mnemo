"""The thin per-agent connector: a stdio MCP proxy into the shared service.

This is what each agent's MCP client launches (`mnemo-mcp`). It loads nothing
heavy — it forwards initialize / list_tools / call_tool to the shared
`mnemo-service` over streamable-http, so the embedder and store live once in
that one process regardless of how many agents are connected.
"""
from __future__ import annotations

from mnemo.adapters.mcp.connector_presence import ConnectorPresence
from mnemo.adapters.mcp.launcher import ensure_service_running
from mnemo.adapters.mcp.run_paths import connectors_dir
from mnemo.adapters.session.in_process_session_provider import InProcessSessionProvider
from mnemo.adapters.session.meta_session_provider import SESSION_META_KEY
from mnemo.infrastructure.config import Config


async def _serve(url: str, session: InProcessSessionProvider, config: Config) -> None:
    import anyio
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    async def _forward(action):
        # Open a fresh upstream connection for this request, (re)starting the shared
        # service first if it is not up. A single long-lived stream would die if the
        # service restarts mid-session (e.g. `mnemo reindex` stops it), leaving the
        # connector wedged; per-request connect + one retry recovers transparently.
        # ensure_service_running blocks (socket waits), so it runs off the event loop.
        last_error: Exception | None = None
        for _attempt in range(2):
            await anyio.to_thread.run_sync(ensure_service_running, config)
            try:
                async with streamable_http_client(url) as (http_read, http_write, _):
                    async with ClientSession(http_read, http_write) as upstream:
                        await upstream.initialize()
                        return await action(upstream)
            except Exception as exc:  # noqa: BLE001 — transport/connection failure: retry once
                last_error = exc
        raise last_error

    server = Server("mnemo")

    @server.list_tools()
    async def list_tools():
        return (await _forward(lambda upstream: upstream.list_tools())).tools

    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        # Forward the upstream result verbatim (content + structured + isError),
        # tagging the request with this connection's session id.
        return await _forward(
            lambda upstream: upstream.call_tool(
                name, arguments, meta={SESSION_META_KEY: session.current_session_id()}
            )
        )

    async with stdio_server() as (stdio_read, stdio_write):
        await server.run(stdio_read, stdio_write, server.create_initialization_options())


def main() -> None:
    import anyio

    config = Config.from_env()
    # This connector is one process per agent, so it owns the run's session id;
    # it travels to the service as request metadata (the service never invents it).
    session = InProcessSessionProvider()
    # Mark this connector alive: hold a flock the kernel frees when we die, so the
    # service can count live connectors and idle-exit once none remain. Held for
    # the whole run (the reference lives until anyio.run returns, i.e. process end).
    presence = ConnectorPresence(connectors_dir(config))
    presence.acquire(session.current_session_id())
    ensure_service_running(config)  # bring the shared service up when this agent connects
    # Each forwarded request re-ensures it too, so a restart mid-session (e.g. `mnemo
    # reindex` stops it) recovers instead of wedging — see _serve._forward.
    anyio.run(_serve, f"http://{config.host}:{config.port}/mcp", session, config)
