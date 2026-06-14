"""Port: wire one MCP client to the mnemo connector.

Each client has its own integration (its official `mcp add` CLI, or a config
file in its own schema), but they share this shape: report whether the client is
present on this machine, describe what wiring it would do, and do it.
"""
from __future__ import annotations

from typing import Protocol

from mnemo.adapters.setup.install_result import InstallResult


class ClientInstaller(Protocol):
    @property
    def name(self) -> str:
        """The client's slug (e.g. 'cursor'), used on the command line."""
        ...

    def detect(self) -> bool:
        """True if this client looks installed on this machine."""
        ...

    def describe(self) -> str:
        """A one-line description of what wiring would do (for --dry-run / prompts)."""
        ...

    def install(self) -> InstallResult:
        """Wire the connector into this client (idempotent)."""
        ...
