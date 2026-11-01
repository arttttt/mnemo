"""Read/merge/write a client's JSON config without disturbing its other keys."""
from __future__ import annotations

import json
from pathlib import Path

from mnemo.adapters.setup.config_file import replace_file


def load_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    replace_file(path, text.encode("utf-8"))
