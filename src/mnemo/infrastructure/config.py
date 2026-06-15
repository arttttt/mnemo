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
        )
