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

    @staticmethod
    def from_env() -> "Config":
        data_dir = os.path.expanduser(os.environ.get("MNEMO_DATA_DIR", "~/.mnemo/data"))
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        return Config(
            data_dir=data_dir,
            embedder=os.environ.get("MNEMO_EMBEDDER", "fastembed"),
            store=os.environ.get("MNEMO_STORE", "memory"),
            store_path=os.environ.get(
                "MNEMO_STORE_PATH", os.path.join(data_dir, "memory.json")
            ),
        )
