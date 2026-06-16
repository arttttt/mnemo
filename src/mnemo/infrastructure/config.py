"""Configuration (composition-root concern). Reads MNEMO_* environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: str
    embedder: str
    store: str
    store_path: str
    embed_model: str | None = None  # concrete fastembed model; None => adapter default
    sqlite_path: str = ""
    host: str = "127.0.0.1"   # the shared service binds localhost-only
    port: int = 8765
    idle_grace_seconds: float = 300.0         # exit this long after the last connector leaves
    idle_check_interval_seconds: float = 5.0  # how often the service sweeps for live connectors
    # Deferred embedding (the service's async worker pool; docs/03-architecture.md).
    embed_workers: int = 1                     # parallel encodes — also the RAM bound (default 1 = safe)
    embed_queue_max: int = 256                 # backlog cap; above it a write embeds synchronously
    embed_max_retries: int = 3                 # retries before a memory is left lexical-only
    embed_drain_timeout: float = 30.0          # how long idle-exit waits for the queue to drain

    @staticmethod
    def from_env() -> "Config":
        data_dir = os.path.expanduser(os.environ.get("MNEMO_DATA_DIR", "~/.mnemo/data"))
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        return Config(
            data_dir=data_dir,
            embedder=os.environ.get("MNEMO_EMBEDDER", "fastembed"),
            embed_model=os.environ.get("MNEMO_EMBED_MODEL") or None,
            store=os.environ.get("MNEMO_STORE", "sqlite"),
            store_path=os.environ.get(
                "MNEMO_STORE_PATH", os.path.join(data_dir, "memory.json")
            ),
            sqlite_path=os.environ.get(
                "MNEMO_SQLITE_PATH", os.path.join(data_dir, "memory.db")
            ),
            host=os.environ.get("MNEMO_HOST", "127.0.0.1"),
            port=int(os.environ.get("MNEMO_PORT", "8765")),
            idle_grace_seconds=float(os.environ.get("MNEMO_IDLE_GRACE_SECONDS", "300")),
            idle_check_interval_seconds=float(
                os.environ.get("MNEMO_IDLE_CHECK_INTERVAL_SECONDS", "5")
            ),
            embed_workers=int(os.environ.get("MNEMO_EMBED_WORKERS", "1")),
            embed_queue_max=int(os.environ.get("MNEMO_EMBED_QUEUE_MAX", "256")),
            embed_max_retries=int(os.environ.get("MNEMO_EMBED_MAX_RETRIES", "3")),
            embed_drain_timeout=float(os.environ.get("MNEMO_EMBED_DRAIN_TIMEOUT", "30")),
        )
