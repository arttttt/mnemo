"""Read/merge/write a client's JSON config without disturbing its other keys."""
from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text())


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
