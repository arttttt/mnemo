"""The set of MCP clients mnemo can wire to, each behind a ClientInstaller.

Two ship an official `mcp add` command (we shell out to it); four are wired by
writing their config file directly. Claude Code, Kimi Code and OpenCode also get
mnemo's permission rules (permission_policy), since they ask before every MCP call
their config does not allow. The connector argv, `home` and the environment are
injectable so the whole set is testable against a temp home without touching real
configs.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from mnemo.adapters.setup.claude_permission_rules import ClaudePermissionRules
from mnemo.adapters.setup.cli_client_installer import CliClientInstaller
from mnemo.adapters.setup.client_installer import ClientInstaller
from mnemo.adapters.setup.command_runner import CommandRunner, SubprocessCommandRunner
from mnemo.adapters.setup.connector_command import connector_command
from mnemo.adapters.setup.kimi_permission_rules import KimiPermissionRules
from mnemo.adapters.setup.mcp_servers_json_installer import McpServersJsonInstaller
from mnemo.adapters.setup.opencode_installer import OpencodeInstaller
from mnemo.adapters.setup.opencode_permission_rules import OpencodePermissionRules
from mnemo.adapters.setup.with_permission_rules import WithPermissionRules

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
    kimi_home = _kimi_home(home, env)
    opencode_json = home / ".config" / "opencode" / "opencode.json"
    return [
        WithPermissionRules(
            CliClientInstaller(
                "claude-code", "claude",
                ["mcp", "add", "--transport", "stdio", "--scope", "user"],
                SERVER_NAME, argv, runner,
            ),
            ClaudePermissionRules(_claude_home(home, env) / "settings.json", SERVER_NAME),
        ),
        # Codex needs no rules: it runs read-only tools and non-destructive, closed-world
        # writes without a prompt, from the tools' annotations, and asks before deletes.
        CliClientInstaller("codex", "codex", ["mcp", "add"], SERVER_NAME, argv, runner),
        # Kimi Code has no `mcp add` command; it reads its servers from mcp.json in its
        # home directory.
        WithPermissionRules(
            McpServersJsonInstaller("kimi-code", kimi_home / "mcp.json", SERVER_NAME, argv),
            KimiPermissionRules(kimi_home / "config.toml", SERVER_NAME),
        ),
        McpServersJsonInstaller(
            "cursor", home / ".cursor" / "mcp.json", SERVER_NAME, argv
        ),
        McpServersJsonInstaller(
            "windsurf", home / ".codeium" / "windsurf" / "mcp_config.json", SERVER_NAME, argv
        ),
        WithPermissionRules(
            OpencodeInstaller("opencode", opencode_json, SERVER_NAME, argv),
            OpencodePermissionRules(opencode_json, SERVER_NAME),
        ),
    ]


def _claude_home(home: Path, env: Mapping[str, str]) -> Path:
    # Claude Code keeps its user settings in $CLAUDE_CONFIG_DIR when set, else in ~/.claude.
    override = env.get("CLAUDE_CONFIG_DIR")
    return Path(override) if override else home / ".claude"


def _kimi_home(home: Path, env: Mapping[str, str]) -> Path:
    # Kimi Code keeps its config in $KIMI_CODE_HOME when set, else in ~/.kimi-code.
    override = env.get("KIMI_CODE_HOME")
    return Path(override) if override else home / ".kimi-code"
