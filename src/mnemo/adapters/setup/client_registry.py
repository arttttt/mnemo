"""The set of MCP clients mnemo can wire to, each behind a ClientInstaller.

Three ship an official `mcp add` command (we shell out to it); three are wired by
writing their config file directly. The connector argv and `home` are injectable
so the whole set is testable against a temp home without touching real configs.
"""
from __future__ import annotations

from pathlib import Path

from mnemo.adapters.setup.cli_client_installer import CliClientInstaller
from mnemo.adapters.setup.client_installer import ClientInstaller
from mnemo.adapters.setup.command_runner import CommandRunner, SubprocessCommandRunner
from mnemo.adapters.setup.connector_command import connector_command
from mnemo.adapters.setup.mcp_servers_json_installer import McpServersJsonInstaller
from mnemo.adapters.setup.opencode_installer import OpencodeInstaller

SERVER_NAME = "mnemo"


def build_installers(
    connector_argv: list[str] | None = None,
    runner: CommandRunner | None = None,
    home: Path | None = None,
) -> list[ClientInstaller]:
    argv = connector_argv if connector_argv is not None else connector_command()
    runner = runner or SubprocessCommandRunner()
    home = home or Path.home()
    return [
        CliClientInstaller(
            "claude-code", "claude",
            ["mcp", "add", "--transport", "stdio", "--scope", "user"],
            SERVER_NAME, argv, runner,
        ),
        CliClientInstaller("codex", "codex", ["mcp", "add"], SERVER_NAME, argv, runner),
        CliClientInstaller("kimi-code", "kimi", ["mcp", "add"], SERVER_NAME, argv, runner),
        McpServersJsonInstaller(
            "cursor", home / ".cursor" / "mcp.json", SERVER_NAME, argv
        ),
        McpServersJsonInstaller(
            "windsurf", home / ".codeium" / "windsurf" / "mcp_config.json", SERVER_NAME, argv
        ),
        OpencodeInstaller(
            "opencode", home / ".config" / "opencode" / "opencode.json", SERVER_NAME, argv
        ),
    ]
