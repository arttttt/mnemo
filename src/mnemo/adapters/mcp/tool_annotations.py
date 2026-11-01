"""How each mnemo tool declares its side effects to MCP clients.

Clients decide from these hints whether a call needs the user's approval and whether it
may run alongside others. Codex skips the approval prompt for a read-only tool, and for
a writing one only when it is declared both non-destructive and closed-world; Claude Code
runs read-only calls in parallel. Nothing mnemo does reaches outside the local store, so
every tool is closed-world. Imports the MCP SDK, so it is loaded lazily with the server.
"""
from mcp.types import ToolAnnotations

# Reads the store and changes nothing.
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)

# Adds or changes, losing no memory: a topic_key update keeps the old version in its
# chain, and update_project only rewrites a project's one-line description.
WRITES = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)

# Removes memories for good.
DELETES = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)
