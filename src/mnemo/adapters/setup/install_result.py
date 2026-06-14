"""The outcome of wiring (or planning to wire) one client to mnemo."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstallResult:
    client: str
    status: str  # "ok" | "skipped" | "failed"
    target: str  # the config file written, or the command run
    message: str = ""
