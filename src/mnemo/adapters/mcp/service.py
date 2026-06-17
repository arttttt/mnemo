"""The shared mnemo service: one process serving all agents over MCP.

It owns the single embedder and store and exposes the tools over streamable-http,
bound to localhost. The thin per-agent connector (`mnemo-mcp`) forwards into it,
so the heavy parts load once regardless of how many agents are connected.

The service also idle-exits: a background monitor watches the connector presence
markers and shuts the process down once none remain (see IdleMonitor), so nothing
is resident when no agent is connected.
"""
from __future__ import annotations

import logging
import os
import threading

from mnemo.adapters.embedding.async_embedding_scheduler import AsyncEmbeddingScheduler
from mnemo.adapters.mcp.connector_liveness import ConnectorLiveness
from mnemo.adapters.mcp.idle_monitor import IdleMonitor
from mnemo.adapters.mcp.run_paths import connectors_dir, run_dir
from mnemo.adapters.mcp.server import build_mcp
from mnemo.adapters.session.meta_session_provider import MetaSessionProvider
from mnemo.application.use_cases.remember_memory import RememberMemory
from mnemo.infrastructure.composition import build_container
from mnemo.infrastructure.config import Config
from mnemo.infrastructure.logging_config import configure_logging

_log = logging.getLogger("mnemo.service")


def main() -> None:
    configure_logging()
    config = Config.from_env()
    _migrate_store(config)
    # The session id is owned by each agent's connector and arrives as request
    # metadata; the service just reads it (see MetaSessionProvider).
    session_provider = MetaSessionProvider()
    container = build_container(config, session_provider=session_provider)
    # The service is long-running, so it embeds OFF the hot path: swap the inline
    # scheduler for the async worker pool and rewire the write use case to it. Recovery
    # is automatic — the workers drain the DB's pending rows on start.
    scheduler = AsyncEmbeddingScheduler(
        container.embedder,
        container.repository,
        workers=config.embed_workers,
        queue_max=config.embed_queue_max,
        max_retries=config.embed_max_retries,
    )
    container.scheduler = scheduler
    container.remember = RememberMemory(container.repository, scheduler, session_provider)
    scheduler.start()
    _log.info(
        "service up: embedder=%s dim=%d workers=%d on %s:%d (encode runs on the embed worker, off the write path)",
        config.embedder, container.embedder.dim, config.embed_workers, config.host, config.port,
    )
    mcp = build_mcp(container, host=config.host, port=config.port)
    _start_idle_monitor(config, scheduler)
    mcp.run(transport="streamable-http")


def _migrate_store(config: Config) -> None:
    # One-off, disposable schema migrations for an existing store, run before the
    # store is opened (DB migration policy). Remove once every live store is migrated.
    if config.store != "sqlite":
        return
    from mnemo.infrastructure.migrations import drop_dedup_columns

    dropped = drop_dedup_columns(config.sqlite_path)
    if dropped:
        _log.info("store migration: dropped legacy columns %s", dropped)


def _start_idle_monitor(config: Config, scheduler: AsyncEmbeddingScheduler) -> None:
    monitor = IdleMonitor(
        ConnectorLiveness(connectors_dir(config)),
        on_idle=lambda: _shutdown(config, scheduler),
        grace_seconds=config.idle_grace_seconds,
        interval_seconds=config.idle_check_interval_seconds,
    )
    threading.Thread(target=monitor.run, name="mnemo-idle-monitor", daemon=True).start()


def _shutdown(config: Config, scheduler: AsyncEmbeddingScheduler) -> None:
    # Fires only when no connector is alive → no in-flight requests and no new writes.
    # Finish embedding whatever is still pending (bounded by a timeout; the rest is
    # recovered on the next start). Committed data is durable in the SQLite WAL.
    scheduler.drain(config.embed_drain_timeout)
    scheduler.stop()
    try:
        (run_dir(config) / "service.pid").unlink()
    except FileNotFoundError:
        pass
    os._exit(0)
