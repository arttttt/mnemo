"""The MCP server behind mnemo's tools: the SDK's MCPServer plus mnemo's tool-call boundary.

Every tool call passes through `MnemoMCPServer.call_tool`, which owns what surrounds it:

- calls run one at a time, in arrival order;
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

import anyio
import mcp.types as types
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

from mnemo.adapters.session.meta_session_provider import SESSION_META_KEY, bind_session

_log = logging.getLogger("mnemo.mcp")


class MnemoMCPServer(MCPServer):
    # One tool call at a time, in arrival order (anyio.Lock is FIFO). The SDK runs each
    # tool on a worker thread, so without this, calls from several agents would overlap —
    # and remember's checks-then-write (exact duplicate, topic_key head, then the insert or
    # supersede) could store a duplicate or fork a topic_key's chain. Waiting happens on
    # the event loop, which stays free to serve connections while a long call (recall)
    # runs. Created on first use, inside the running loop. Not reentrant: a tool must
    # never call call_tool itself.
    _call_slot: anyio.Lock | None = None

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Context | None = None
    ) -> types.CallToolResult | types.InputRequiredResult:
        if self._call_slot is None:
            self._call_slot = anyio.Lock()
        async with self._call_slot:
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
