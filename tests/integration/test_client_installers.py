"""The per-client installers at the real boundary.

The CLI-based installers (claude-code/codex) shell out to the client's own binary, so
they are exercised with a fake runner that records the argv. The file-based installers
(kimi-code/cursor/windsurf/opencode) write to a temp home and are checked for an
idempotent, non-destructive upsert.
"""
import json

from mnemo.adapters.setup.cli_client_installer import CliClientInstaller
from mnemo.adapters.setup.client_registry import build_installers
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


def _registered(name, runner, home, env):
    installers = build_installers(connector_argv=_ARGV, runner=runner, home=home, env=env)
    return next(installer for installer in installers if installer.name == name)


def test_kimi_code_is_wired_through_its_mcp_json_not_a_cli(tmp_path):
    # Kimi Code has no `kimi mcp add`; its servers live in ~/.kimi-code/mcp.json. An entry
    # written by hand is replaced, and other servers stay.
    config = tmp_path / ".kimi-code" / "mcp.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"mcpServers": {
        "mnemo": {"type": "stdio", "command": "uv", "args": ["run", "mnemo-mcp"]},
        "other": {"command": "x"},
    }}))
    runner = _FakeRunner()
    kimi = _registered("kimi-code", runner, tmp_path, env={})

    assert kimi.detect()
    result = kimi.install()

    # The status also covers the permission rules, which need tomllib (Python 3.11+);
    # they have their own tests.
    assert result.target.startswith(f"{config}; ")
    assert runner.calls == []
    servers = json.loads(config.read_text())["mcpServers"]
    assert servers == {
        "mnemo": {"command": "/opt/bin/mnemo-mcp", "args": []},
        "other": {"command": "x"},
    }


def test_kimi_code_config_follows_kimi_code_home(tmp_path):
    custom = tmp_path / "kimi-home"
    kimi = _registered("kimi-code", _FakeRunner(), tmp_path, env={"KIMI_CODE_HOME": str(custom)})

    kimi.install()

    assert (custom / "mcp.json").exists()
    assert not (tmp_path / ".kimi-code").exists()


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


def test_subprocess_runner_returns_nonzero_for_a_missing_binary():
    # The named `mnemo setup <client>` path runs the client's binary without detect(),
    # so a missing binary must surface as a non-zero code, not a raised OSError/traceback.
    from mnemo.adapters.setup.command_runner import SubprocessCommandRunner

    code = SubprocessCommandRunner().run(["mnemo-no-such-client-binary-zzqqx"])
    assert code != 0


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


def test_opencode_installer_follows_a_file_in_opencode_2_shape(tmp_path):
    # OpenCode 2 keeps servers under mcp.servers; a version-1 entry of the same name
    # would conflict with the new one, so it goes.
    config = tmp_path / "opencode.json"
    remote = {"type": "remote", "url": "https://example.com/mcp"}
    config.write_text(json.dumps({"mcp": {
        "mnemo": {"type": "local", "command": ["uv", "run", "mnemo-mcp"], "enabled": True},
        "servers": {"other": remote},
    }}))

    OpencodeInstaller("opencode", config, "mnemo", ["/opt/bin/mnemo-mcp"]).install()

    assert json.loads(config.read_text())["mcp"] == {"servers": {
        "other": remote,
        "mnemo": {"type": "local", "command": ["/opt/bin/mnemo-mcp"]},
    }}


def test_opencode_installer_tells_a_version_1_server_named_servers_from_the_map(tmp_path):
    config = tmp_path / "opencode.json"
    named_servers = {"type": "local", "command": ["other-mcp"], "enabled": True}
    config.write_text(json.dumps({"mcp": {"servers": named_servers}}))

    OpencodeInstaller("opencode", config, "mnemo", ["/opt/bin/mnemo-mcp"]).install()

    assert json.loads(config.read_text())["mcp"] == {
        "servers": named_servers,
        "mnemo": {"type": "local", "command": ["/opt/bin/mnemo-mcp"], "enabled": True},
    }
