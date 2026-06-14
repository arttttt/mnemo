"""The per-client installers at the real boundary.

The CLI-based installers (claude-code/codex/kimi-code) shell out to the client's
own binary, so they are exercised with a fake runner that records the argv. The
file-based installers (cursor/windsurf/opencode) write to a temp home and are
checked for an idempotent, non-destructive upsert.
"""
import json

from mnemo.adapters.setup.cli_client_installer import CliClientInstaller
from mnemo.adapters.setup.mcp_servers_json_installer import McpServersJsonInstaller
from mnemo.adapters.setup.opencode_installer import OpencodeInstaller

_ARGV = ["/opt/bin/mnemo-mcp"]


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.code = 0

    def run(self, argv: list[str]) -> int:
        self.calls.append(argv)
        return self.code


def test_cli_installer_runs_the_official_add_command():
    runner = _FakeRunner()
    installer = CliClientInstaller(
        "claude-code", "claude",
        ["mcp", "add", "--transport", "stdio", "--scope", "user"],
        "mnemo", _ARGV, runner,
    )
    result = installer.install()
    assert result.status == "ok"
    assert runner.calls == [
        ["claude", "mcp", "add", "--transport", "stdio", "--scope", "user",
         "mnemo", "--", "/opt/bin/mnemo-mcp"]
    ]


def test_cli_installer_reports_failure_on_nonzero_exit():
    runner = _FakeRunner()
    runner.code = 3
    installer = CliClientInstaller("codex", "codex", ["mcp", "add"], "mnemo", _ARGV, runner)
    result = installer.install()
    assert result.status == "failed" and "3" in result.message


def test_mcp_servers_json_creates_and_preserves_other_servers(tmp_path):
    config = tmp_path / ".cursor" / "mcp.json"
    installer = McpServersJsonInstaller("cursor", config, "mnemo", ["/opt/bin/mnemo-mcp", "--x"])

    installer.install()
    data = json.loads(config.read_text())
    assert data["mcpServers"]["mnemo"] == {"command": "/opt/bin/mnemo-mcp", "args": ["--x"]}

    # A pre-existing, unrelated server must survive a (re-)install.
    config.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    installer.install()
    data = json.loads(config.read_text())
    assert "other" in data["mcpServers"] and "mnemo" in data["mcpServers"]


def test_opencode_installer_uses_local_schema_and_preserves_keys(tmp_path):
    config = tmp_path / ".config" / "opencode" / "opencode.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"$schema": "https://opencode.ai/config.json"}))

    OpencodeInstaller("opencode", config, "mnemo", ["/opt/bin/mnemo-mcp"]).install()
    data = json.loads(config.read_text())
    assert data["$schema"] == "https://opencode.ai/config.json"  # preserved
    assert data["mcp"]["mnemo"] == {
        "type": "local",
        "command": ["/opt/bin/mnemo-mcp"],
        "enabled": True,
    }
