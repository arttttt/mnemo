"""The MCP server behind mnemo's tools: the SDK's MCPServer plus mnemo's tool-call boundary.

A tool registered from a function without a description of its own is described by
its docstring in clean form (inspect.cleandoc). The SDK sends the docstring as is:
Python 3.10–3.12 keep its indentation, and 3.13 strips that at compile time but keeps
the trailing newline. Cleaned, the text is the same whatever Python runs the service.

Every tool call passes through `MnemoMCPServer.call_tool`, which owns what surrounds it:

- calls run one at a time, in arrival order;
- the calling agent's session id — sent by the connector as request `_meta` — is bound
  for the duration of the call, so the write path stamps it on what it stores;
- a failure reaches the agent as "Error executing tool <name>: <message>" whatever the
  exception. The SDK alone shows the message only for its own ToolError and replaces
  anything else with a bare "Error executing tool <name>"; mnemo's use cases signal the
  caller's mistakes (unknown project, scope without project, over-cap content) with
  ordinary exceptions whose message is the guidance the agent needs;
- an answer goes out as structured content only. For backwards compatibility the SDK
  also renders a structured result as text blocks (one per item of a list), and a client
  that shows both, such as Kimi Code, then puts every result in front of the model twice.
  Every client mnemo targets reads structured content when no text is sent, so the text
  copy is dropped. An error result keeps its text, which is all it has.

Imported by `build_mcp` only, so importing the server module stays free of the SDK.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

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

    def add_tool(
        self,
        fn: Callable[..., Any],
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        annotations: types.ToolAnnotations | None = None,
        icons: list[types.Icon] | None = None,
        meta: dict[str, Any] | None = None,
        structured_output: bool | None = None,
    ) -> None:
        if not description and fn.__doc__:
            description = inspect.cleandoc(fn.__doc__)
        super().add_tool(
            fn,
            name=name,
            title=title,
            description=description,
            annotations=annotations,
            icons=icons,
            meta=meta,
            structured_output=structured_output,
        )

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Context | None = None
    ) -> types.CallToolResult | types.InputRequiredResult:
        if self._call_slot is None:
            self._call_slot = anyio.Lock()
        async with self._call_slot:
            with bind_session(_session_id(context)):
                try:
                    result = await super().call_tool(name, arguments, context)
                except UnexpectedToolError as exc:
                    cause = exc.__cause__ or exc
                    _log.info("tool %s failed", name, exc_info=cause)
                    raise ToolError(f"Error executing tool {name}: {cause}") from exc
        return _structured_only(result)


def _structured_only(
    result: types.CallToolResult | types.InputRequiredResult,
) -> types.CallToolResult | types.InputRequiredResult:
    """The result without the SDK's text rendering of its structured content."""
    if (
        isinstance(result, types.CallToolResult)
        and result.structured_content is not None
        and not result.is_error
    ):
        return result.model_copy(update={"content": []})
    return result


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
