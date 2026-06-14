"""Run an external command (e.g. a client's own `mcp add`).

Behind a port so the CLI-based installers can be tested without the real client
binary — a fake runner records the argv instead of executing it.
"""
from __future__ import annotations

import subprocess
from typing import Protocol


class CommandRunner(Protocol):
    def run(self, argv: list[str]) -> int:
        """Run argv; return its exit code (non-zero on failure)."""
        ...


class SubprocessCommandRunner:
    def run(self, argv: list[str]) -> int:
        return subprocess.run(argv, check=False).returncode
