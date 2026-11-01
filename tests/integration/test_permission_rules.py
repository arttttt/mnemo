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
from mnemo.adapters.setup.opencode_permission_rules import ALREADY_ALLOWED, OpencodePermissionRules
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


# --- OpenCode: opencode.json -----------------------------------------------------------

_OPENCODE_KEYS = [("mnemo_*", "allow"), ("mnemo_delete*", "ask")]


def test_opencode_rules_allow_every_tool_and_ask_before_deletes(tmp_path):
    config = tmp_path / "opencode.json"

    note = OpencodePermissionRules(config, "mnemo").apply()

    assert note == ADDED
    assert list(json.loads(config.read_text())["permission"].items()) == _OPENCODE_KEYS


def test_opencode_rules_go_last_and_keep_everything_else(tmp_path):
    # OpenCode applies the last matching key, so mnemo's keys must follow a broader one
    # such as "*", and the ask rule for deletes must follow the allow rule.
    config = tmp_path / "opencode.json"
    config.write_text(json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "mcp": {"mnemo": {"type": "local", "command": ["mnemo-mcp"], "enabled": True}},
        "permission": {"mnemo_delete*": "ask", "edit": "ask", "*": "ask", "mnemo_*": "allow"},
    }))

    assert OpencodePermissionRules(config, "mnemo").apply() == ADDED

    data = json.loads(config.read_text())
    assert data["$schema"] == "https://opencode.ai/config.json"
    assert data["mcp"] == {"mnemo": {"type": "local", "command": ["mnemo-mcp"], "enabled": True}}
    assert list(data["permission"].items()) == [("edit", "ask"), ("*", "ask"), *_OPENCODE_KEYS]


@pytest.mark.parametrize("word", ["ask", "deny"])
def test_opencode_rules_spell_out_a_one_word_permission(tmp_path, word):
    # OpenCode reads `"permission": "ask"` as `{"*": "ask"}`.
    config = tmp_path / "opencode.json"
    config.write_text(json.dumps({"permission": word}))

    OpencodePermissionRules(config, "mnemo").apply()

    assert list(json.loads(config.read_text())["permission"].items()) == [
        ("*", word), *_OPENCODE_KEYS
    ]


def test_opencode_rules_leave_a_permission_that_allows_everything(tmp_path):
    config = tmp_path / "opencode.json"
    config.write_text(json.dumps({"permission": "allow"}))
    before = _snapshot(config)

    assert OpencodePermissionRules(config, "mnemo").apply() == ALREADY_ALLOWED
    assert _snapshot(config) == before


def test_opencode_rules_already_in_place_leave_the_file_untouched(tmp_path):
    config = tmp_path / "opencode.json"
    rules = OpencodePermissionRules(config, "mnemo")
    rules.apply()
    before = _snapshot(config)

    assert rules.apply() == PRESENT
    assert _snapshot(config) == before


@pytest.mark.parametrize("content", [
    "{not json",
    json.dumps({"permission": ["mnemo_*"]}),
    json.dumps({"permission": "sometimes"}),
])
def test_opencode_rules_refuse_a_config_they_cannot_change_safely(tmp_path, content):
    config = tmp_path / "opencode.json"
    config.write_text(content)

    with pytest.raises(RulesNotWritten) as refused:
        OpencodePermissionRules(config, "mnemo").apply()

    assert config.read_text() == content
    assert list(json.loads(refused.value.snippet)["permission"].items()) == _OPENCODE_KEYS


def test_opencode_setup_wires_the_server_and_its_rules_in_one_file(tmp_path):
    opencode = next(
        installer
        for installer in build_installers(
            connector_argv=["/opt/bin/mnemo-mcp"], runner=_Runner(0), home=tmp_path, env={},
        )
        if installer.name == "opencode"
    )

    result = opencode.install()

    config = tmp_path / ".config" / "opencode" / "opencode.json"
    assert result.status == "ok" and result.message == ADDED
    data = json.loads(config.read_text())
    assert data["mcp"]["mnemo"]["command"] == ["/opt/bin/mnemo-mcp"]
    assert list(data["permission"].items()) == _OPENCODE_KEYS


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
