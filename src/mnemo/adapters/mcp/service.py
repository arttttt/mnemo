"""The shared mnemo service: one process serving all agents over MCP.

It owns the single embedder and store and exposes the tools over streamable-http,
bound to localhost. The thin per-agent connector (`mnemo-mcp`) forwards into it,
so the heavy parts load once regardless of how many agents are connected.
"""
from __future__ import annotations

from mnemo.adapters.mcp.server import build_mcp
from mnemo.infrastructure.composition import build_container
from mnemo.infrastructure.config import Config


def main() -> None:
    config = Config.from_env()
    mcp = build_mcp(build_container(config), host=config.host, port=config.port)
    mcp.run(transport="streamable-http")
