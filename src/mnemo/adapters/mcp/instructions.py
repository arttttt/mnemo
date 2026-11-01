"""What mnemo tells an agent about itself when the agent connects.

These are the MCP server instructions. Clients that load tools on demand (Claude Code,
Codex) show the model little more than tool names and this text up front, so it says
WHEN to reach for mnemo; how each tool works stays in that tool's own description.

The per-agent proxy answers `initialize` itself, so it is the one that must send them
(the service's own would never reach the agent); the service sends the same text to
clients that connect to it directly. Kept free of the MCP SDK so the proxy can import
it without loading anything heavy.
"""

SERVER_INSTRUCTIONS = (
    "mnemo is the long-term memory shared by all of the user's coding agents: what one "
    "agent saves is available to every other agent and to future sessions.\n"
    "\n"
    "- Before researching, designing or deciding something non-trivial, look for earlier "
    "work on the topic with `search`; another agent may have done it already. Use `get` "
    "for a memory's full record and its version history.\n"
    "- When you reach a decision, finding, rule or checkpoint worth keeping, save it with "
    "`remember`: one topic per memory, with a stable `topic_key` so a later update "
    "replaces it instead of adding a duplicate.\n"
    "- Memories live in registered projects (`list_projects`): pass the project's slug, "
    "usually the repository's name, as `project`. Use `scope=\"global\"` only for knowledge "
    "that applies across projects.\n"
    "- Never store secrets such as keys, tokens or passwords."
)
