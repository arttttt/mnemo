"""The MCP server behind mnemo's tools: the SDK's MCPServer plus mnemo's tool-call boundary.

Every tool call passes through `MnemoMCPServer.call_tool`, which owns what surrounds it:

- the calling agent's session id — sent by the connector as request `_meta` — is bound
  for the duration of the call, so the write path stamps it on what it stores;
- a failure reaches the agent as "Error executing tool <name>: <message>" whatever the
  exception. The SDK alone shows the message only for its own ToolError and replaces
  anything else with a bare "Error executing tool <name>"; mnemo's use cases signal the
  caller's mistakes (unknown project, scope without project, over-cap content) with
  ordinary exceptions whose message is the guidance the agent needs.

Imported by `build_mcp` only, so importing the server module stays free of the SDK.
"""
from __future__ import annotations

import logging
from typing import Any

import mcp.types as types
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

from mnemo.adapters.session.meta_session_provider import SESSION_META_KEY, bind_session

_log = logging.getLogger("mnemo.mcp")


class MnemoMCPServer(MCPServer):
    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Context | None = None
    ) -> types.CallToolResult | types.InputRequiredResult:
        with bind_session(_session_id(context)):
            try:
                return await super().call_tool(name, arguments, context)
            except UnexpectedToolError as exc:
                cause = exc.__cause__ or exc
                _log.info("tool %s failed", name, exc_info=cause)
                raise ToolError(f"Error executing tool {name}: {cause}") from exc


def _session_id(context: Context | None) -> str | None:
    """The session id the connector sent with this request; None for a call made outside
    a request (in-process, with or without a Context) or one that carried no `_meta`."""
    if context is None:
        return None
    try:
        meta = context.request_context.meta
    except ValueError:  # a Context built outside a request has no request context
        return None
    return (meta or {}).get(SESSION_META_KEY)
