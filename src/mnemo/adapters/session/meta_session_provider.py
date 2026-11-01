"""Service-side session provider: reads the session id supplied by the client.

In the shared service the session id is owned by the per-agent connector (the
`mnemo-mcp` proxy), which generates one id per run and sends it on every request
as MCP `_meta`. The MCP server binds that id for the duration of the tool call
(`bind_session`); this provider just reads the bound id — the service never invents
one. The binding is a context variable, so it follows the call onto the worker
thread that runs the tool. Liskov-substitutable behind SessionProvider; returns None
when no id was supplied (e.g. a direct, non-proxy client).
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

SESSION_META_KEY = "mnemo_session_id"

_current_session_id: ContextVar[str | None] = ContextVar("mnemo_session_id", default=None)


@contextmanager
def bind_session(session_id: str | None) -> Iterator[None]:
    """Make ``session_id`` the current session for everything run inside the block."""
    token = _current_session_id.set(session_id)
    try:
        yield
    finally:
        _current_session_id.reset(token)


class MetaSessionProvider:
    def current_session_id(self) -> str | None:
        return _current_session_id.get()
