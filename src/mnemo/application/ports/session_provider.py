"""Port: supplies the current run's session id (write-time provenance)."""
from __future__ import annotations

from typing import Protocol


class SessionProvider(Protocol):
    def current_session_id(self) -> str | None: ...
