"""Wire a client that uses the `mcpServers` JSON schema (Cursor, Windsurf).

`{ "mcpServers": { "<name>": { "command": "...", "args": [...] } } }`. The entry
is upserted: other servers and unrelated keys in the file are preserved.
"""
from __future__ import annotations

from pathlib import Path

from mnemo.adapters.setup.install_result import InstallResult
from mnemo.adapters.setup.json_config import load_json, save_json


class McpServersJsonInstaller:
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
        servers = data.setdefault("mcpServers", {})
        servers[self._server_name] = {
            "command": self._connector_argv[0],
            "args": self._connector_argv[1:],
        }
        save_json(self._config_path, data)
        return InstallResult(self._name, "ok", str(self._config_path))
