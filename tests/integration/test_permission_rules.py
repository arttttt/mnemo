"""Permission rules written into each client's own config, against a temp home.

Every adapter must add mnemo's rules, keep the rest of the file, leave the file
untouched when the rules are already there, and write nothing to a file it cannot
safely change.
"""
import json

import pytest

from mnemo.adapters.setup.claude_permission_rules import ClaudePermissionRules
from mnemo.adapters.setup.client_registry import build_installers
from mnemo.adapters.setup.install_result import InstallResult
from mnemo.adapters.setup.permission_rules import ADDED, PRESENT, RulesNotWritten
from mnemo.adapters.setup.with_permission_rules import WithPermissionRules


def _snapshot(path):
    return path.read_bytes(), path.stat().st_mtime_ns


# --- Claude Code: ~/.claude/settings.json -------------------------------------------


def test_claude_rules_allow_every_tool_and_ask_before_deletes(tmp_path):
    settings = tmp_path / ".claude" / "settings.json"

    note = ClaudePermissionRules(settings, "mnemo").apply()

    assert note == ADDED
    assert json.loads(settings.read_text()) == {"permissions": {
        "allow": ["mcp__mnemo__*"],
        "ask": ["mcp__mnemo__delete*"],
    }}


def test_claude_rules_keep_the_other_settings_and_rules(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "model": "opus",
        "permissions": {"allow": ["Bash(git status)"], "deny": ["Read(./.env)"]},
    }))

    ClaudePermissionRules(settings, "mnemo").apply()

    assert json.loads(settings.read_text()) == {
        "model": "opus",
        "permissions": {
            "allow": ["Bash(git status)", "mcp__mnemo__*"],
            "deny": ["Read(./.env)"],
            "ask": ["mcp__mnemo__delete*"],
        },
    }


def test_claude_rules_already_in_place_leave_the_file_untouched(tmp_path):
    settings = tmp_path / "settings.json"
    rules = ClaudePermissionRules(settings, "mnemo")
    rules.apply()
    before = _snapshot(settings)

    assert rules.apply() == PRESENT
    assert _snapshot(settings) == before


def test_claude_rules_add_only_the_missing_one(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"permissions": {"allow": ["mcp__mnemo__*"]}}))

    assert ClaudePermissionRules(settings, "mnemo").apply() == ADDED
    assert json.loads(settings.read_text())["permissions"] == {
        "allow": ["mcp__mnemo__*"],
        "ask": ["mcp__mnemo__delete*"],
    }


@pytest.mark.parametrize("content", [
    "{not json",
    "[]",
    json.dumps({"permissions": ["mcp__mnemo__*"]}),
    json.dumps({"permissions": {"ask": "mcp__mnemo__delete*"}}),
])
def test_claude_rules_refuse_settings_they_cannot_change_safely(tmp_path, content):
    settings = tmp_path / "settings.json"
    settings.write_text(content)

    with pytest.raises(RulesNotWritten) as refused:
        ClaudePermissionRules(settings, "mnemo").apply()

    assert settings.read_text() == content
    assert json.loads(refused.value.snippet) == {"permissions": {
        "allow": ["mcp__mnemo__*"],
        "ask": ["mcp__mnemo__delete*"],
    }}


def test_claude_settings_follow_claude_config_dir(tmp_path):
    custom = tmp_path / "claude-config"
    claude = next(
        installer
        for installer in build_installers(
            connector_argv=["/opt/bin/mnemo-mcp"], runner=_Runner(0), home=tmp_path,
            env={"CLAUDE_CONFIG_DIR": str(custom)},
        )
        if installer.name == "claude-code"
    )

    claude.install()

    assert (custom / "settings.json").exists()
    assert not (tmp_path / ".claude").exists()


# --- Wiring a server and writing its rules together ----------------------------------


class _Runner:
    def __init__(self, code):
        self.code = code

    def run(self, argv):
        return self.code


class _Installer:
    def __init__(self, status, message=""):
        self._result = InstallResult("claude-code", status, "claude mcp add mnemo", message)

    name = "claude-code"

    def detect(self):
        return True

    def describe(self):
        return "run `claude mcp add mnemo`"

    def install(self):
        return self._result


def test_rules_are_written_even_when_wiring_the_server_fails(tmp_path):
    # `claude mcp add` refuses a server that is already there; a re-run still adds the rules.
    settings = tmp_path / "settings.json"
    installer = WithPermissionRules(
        _Installer("failed", "`claude` exited with 1"), ClaudePermissionRules(settings, "mnemo")
    )

    result = installer.install()

    assert result.status == "failed"
    assert result.target == f"claude mcp add mnemo; {settings}"
    assert result.message == f"`claude` exited with 1; {ADDED}"
    assert "mcp__mnemo__*" in json.loads(settings.read_text())["permissions"]["allow"]


def test_rules_that_cannot_be_written_fail_the_install_and_say_what_to_add(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text("{not json")
    installer = WithPermissionRules(_Installer("ok"), ClaudePermissionRules(settings, "mnemo"))

    result = installer.install()

    assert result.status == "failed"
    assert str(settings) in result.message and "mcp__mnemo__delete*" in result.message
    assert settings.read_text() == "{not json"


def test_the_plan_names_the_server_command_and_the_rules(tmp_path):
    settings = tmp_path / "settings.json"
    installer = WithPermissionRules(_Installer("ok"), ClaudePermissionRules(settings, "mnemo"))

    plan = installer.describe()

    assert plan.startswith("run `claude mcp add mnemo`; ")
    assert "allow mcp__mnemo__*" in plan and "ask mcp__mnemo__delete*" in plan
    assert not settings.exists()
