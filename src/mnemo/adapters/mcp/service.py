"""The shared mnemo service: one process serving all agents over MCP.

It owns the single embedder and store and exposes the tools over streamable-http,
bound to localhost. The thin per-agent connector (`mnemo-mcp`) forwards into it,
so the heavy parts load once regardless of how many agents are connected.

The service also idle-exits: a background monitor watches the connector presence
markers and shuts the process down once none remain (see IdleMonitor), so nothing
is resident when no agent is connected.
"""
from __future__ import annotations

import os
import threading

from mnemo.adapters.mcp.connector_liveness import ConnectorLiveness
from mnemo.adapters.mcp.idle_monitor import IdleMonitor
from mnemo.adapters.mcp.run_paths import connectors_dir, run_dir
from mnemo.adapters.mcp.server import build_mcp
from mnemo.adapters.session.meta_session_provider import MetaSessionProvider
from mnemo.infrastructure.composition import build_container
from mnemo.infrastructure.config import Config


def main() -> None:
    config = Config.from_env()
    # The session id is owned by each agent's connector and arrives as request
    # metadata; the service just reads it (see MetaSessionProvider).
    container = build_container(config, session_provider=MetaSessionProvider())
    mcp = build_mcp(container, host=config.host, port=config.port)
    _start_idle_monitor(config)
    mcp.run(transport="streamable-http")


def _start_idle_monitor(config: Config) -> None:
    monitor = IdleMonitor(
        ConnectorLiveness(connectors_dir(config)),
        on_idle=lambda: _shutdown(config),
        grace_seconds=config.idle_grace_seconds,
        interval_seconds=config.idle_check_interval_seconds,
    )
    threading.Thread(target=monitor.run, name="mnemo-idle-monitor", daemon=True).start()


def _shutdown(config: Config) -> None:
    # Fires only when no connector is alive, so there are no in-flight requests.
    # Committed data is durable in the SQLite WAL; just drop the pidfile and exit.
    try:
        (run_dir(config) / "service.pid").unlink()
    except FileNotFoundError:
        pass
    os._exit(0)
