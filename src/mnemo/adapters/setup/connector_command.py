"""Resolve the argv that launches the mnemo connector (`mnemo-mcp`).

Returned as a full argv list so every client formats it its own way (a CLI's
`-- cmd args`, or a `command`/`args` split, or a `command` array). We prefer an
ABSOLUTE path: GUI clients (Cursor, Windsurf) launch servers with their own PATH,
not the shell's, so a bare `mnemo-mcp` may not resolve. When the tool is not on
PATH (running from a source checkout) we fall back to `uv run --directory <repo>`.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def connector_command() -> list[str]:
    found = shutil.which("mnemo-mcp")
    if found:
        return [found]
    uv = shutil.which("uv") or "uv"
    return [uv, "run", "--directory", str(_repo_root()), "mnemo-mcp"]


def _repo_root() -> Path:
    # Walk up from this file until a pyproject.toml is found (the checkout root).
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path(__file__).resolve().parents[4]  # src/mnemo/adapters/setup -> repo
