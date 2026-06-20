"""Service-side session provider: reads the session id supplied by the client.

In the shared service the session id is owned by the per-agent connector (the
`mnemo-mcp` proxy), which generates one id per run and sends it on every request
as MCP `_meta`. This provider just reads that id from the request context — the
service never invents one. Liskov-substitutable behind SessionProvider;
returns None when no id was supplied (e.g. a direct, non-proxy client).
"""
from __future__ import annotations

SESSION_META_KEY = "mnemo_session_id"


class MetaSessionProvider:
    def current_session_id(self) -> str | None:
        from mcp.server.lowlevel.server import request_ctx

        context = request_ctx.get(None)
        if context is None or context.meta is None:
            return None
        return getattr(context.meta, SESSION_META_KEY, None)
