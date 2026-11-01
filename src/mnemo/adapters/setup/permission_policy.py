"""The permission policy setup writes into every client that needs config rules for it.

mnemo's tools run without an approval prompt, except the ones that delete memories,
which still ask. Codex reaches this from the tools' annotations alone. Claude Code,
Kimi Code and OpenCode ask before every MCP call unless their config allows it, so
setup writes these rules into their config files.

Each rule names tools by a glob over mnemo's own tool names; each client's adapter
turns that glob into the client's name for the tool (`mcp__mnemo__delete*` in Claude
Code and Kimi Code, `mnemo_delete*` in OpenCode).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PermissionRule:
    decision: str  # "allow" | "ask"
    tools: str  # glob over the server's tool names


POLICY = (
    PermissionRule("allow", "*"),
    PermissionRule("ask", "delete*"),
)
