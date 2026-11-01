"""The set of MCP clients mnemo can wire to, each behind a ClientInstaller.

Two ship an official `mcp add` command (we shell out to it); four are wired by
writing their config file directly. The connector argv, `home` and the environment
are injectable so the whole set is testable against a temp home without touching
real configs.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
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
    env: Mapping[str, str] | None = None,
) -> list[ClientInstaller]:
    argv = connector_argv if connector_argv is not None else connector_command()
    runner = runner or SubprocessCommandRunner()
    home = home or Path.home()
    env = os.environ if env is None else env
    return [
        CliClientInstaller(
            "claude-code", "claude",
            ["mcp", "add", "--transport", "stdio", "--scope", "user"],
            SERVER_NAME, argv, runner,
        ),
        CliClientInstaller("codex", "codex", ["mcp", "add"], SERVER_NAME, argv, runner),
        # Kimi Code has no `mcp add` command; it reads its servers from mcp.json in its
        # home directory.
        McpServersJsonInstaller(
            "kimi-code", _kimi_home(home, env) / "mcp.json", SERVER_NAME, argv
        ),
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


def _kimi_home(home: Path, env: Mapping[str, str]) -> Path:
    # Kimi Code keeps its config in $KIMI_CODE_HOME when set, else in ~/.kimi-code.
    override = env.get("KIMI_CODE_HOME")
    return Path(override) if override else home / ".kimi-code"
