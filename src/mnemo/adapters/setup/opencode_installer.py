"""Wire OpenCode, which uses its own `mcp` schema (not `mcpServers`).

OpenCode 1 reads `{ "mcp": { "<name>": { "type": "local", "command": [...], "enabled":
true } } }` — `command` is the full argv ARRAY (no separate args field). OpenCode 2
reads that too, converting it in memory without rewriting the file, so it stays the
default. A file already in OpenCode 2's own shape, with servers under `mcp.servers`,
gets `mcp.servers.<name> = { "type": "local", "command": [...] }` instead, and a
version-1 entry of the same name is dropped, since OpenCode 2 would report the two as
a conflict. The entry is upserted; other servers and keys (e.g. `$schema`) are kept.
"""
from __future__ import annotations

from pathlib import Path

from mnemo.adapters.setup.install_result import InstallResult
from mnemo.adapters.setup.json_config import load_json, save_json


class OpencodeInstaller:
    def __init__(
        self, name: str, config_path: Path, server_name: str, connector_argv: list[str]
    ) -> None:
        self._name = name
        self._config_path = Path(config_path)
        self._server_name = server_name
        self._connector_argv = connector_argv

    @property
    def name(self) -> str:
        return self._name

    def detect(self) -> bool:
        return self._config_path.parent.exists()

    def describe(self) -> str:
        return f"write `{self._server_name}` into {self._config_path}"

    def install(self) -> InstallResult:
        data = load_json(self._config_path)
        mcp = data.setdefault("mcp", {})
        entry = {"type": "local", "command": list(self._connector_argv)}
        servers = mcp.get("servers")
        # A version-1 server could itself be named "servers"; OpenCode 2 tells it from
        # its own map by the entry's `type`.
        if isinstance(servers, dict) and servers.get("type") not in ("local", "remote"):
            mcp.pop(self._server_name, None)
            servers[self._server_name] = entry
        else:
            mcp[self._server_name] = {**entry, "enabled": True}
        save_json(self._config_path, data)
        return InstallResult(self._name, "ok", str(self._config_path))
