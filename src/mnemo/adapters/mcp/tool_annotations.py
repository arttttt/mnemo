"""Hints mnemo's tools give MCP clients beyond their schemas: side effects and loading.

Clients decide from the side-effect hints whether a call needs the user's approval and
whether it may run alongside others. Codex skips the approval prompt for a read-only
tool, and for a writing one only when it is declared both non-destructive and
closed-world; Claude Code runs read-only calls in parallel. Every tool is closed-world:
it works on the local store and sends no memory content anywhere (the only network
access is the first download of the local models). Imports the MCP SDK, so it is loaded
lazily with the server.
"""
from mcp.types import ToolAnnotations

# Reads the store and changes nothing.
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)

# Adds or changes, losing no memory: a topic_key update keeps the old version in its
# chain. update_project does overwrite the previous description, which is not kept;
# declared non-destructive anyway, since it is one line that is easy to set again and no
# memory is touched.
WRITES = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)

# Removes memories for good.
DELETES = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)

# Tool _meta: Claude Code loads a tool marked this way at session start instead of
# deferring it behind tool search. For search only — the tool an agent should reach for
# before starting work; every tool loaded up front costs context in every session.
LOAD_UP_FRONT = {"anthropic/alwaysLoad": True}
