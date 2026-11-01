"""Kimi Code's permission rules, appended to config.toml as text.

The rest of the file must stay byte for byte, the rules must read back as the tables
Kimi Code parses, and a file that a new table cannot extend must be left alone.
Reading TOML needs tomllib (Python 3.11+); the tests that parse it skip on 3.10, where
the adapter writes nothing and prints the rules instead, which is tested on every version.
"""
import json
import sys

import pytest

from mnemo.adapters.setup import kimi_permission_rules
from mnemo.adapters.setup.client_registry import build_installers
from mnemo.adapters.setup.kimi_permission_rules import KimiPermissionRules
from mnemo.adapters.setup.permission_rules import ADDED, PRESENT, RulesNotWritten

needs_tomllib = pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib is 3.11+")

_RULES = [
    {"decision": "allow", "pattern": "mcp__mnemo__*"},
    {"decision": "ask", "pattern": "mcp__mnemo__delete*"},
]


def _rules_in(path):
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8"))["permission"]["rules"]


@needs_tomllib
def test_kimi_rules_allow_every_tool_and_ask_before_deletes(tmp_path):
    config = tmp_path / ".kimi-code" / "config.toml"

    note = KimiPermissionRules(config, "mnemo").apply()

    assert note == ADDED
    assert _rules_in(config) == _RULES


@needs_tomllib
def test_kimi_rules_are_appended_and_the_file_before_them_stays_byte_for_byte(tmp_path):
    config = tmp_path / "config.toml"
    original = (
        "# my Kimi settings\n"
        'default_model = "kimi-k2"\n'
        "\n"
        "[[permission.rules]]\n"
        'decision = "allow"\n'
        'pattern = "Shell(git status)"\n'
        "\n"
        "[ui]\n"
        'theme = "dark"'  # no trailing newline
    )
    config.write_text(original, encoding="utf-8")

    KimiPermissionRules(config, "mnemo").apply()

    updated = config.read_text(encoding="utf-8")
    assert updated.startswith(original + "\n")
    assert _rules_in(config) == [
        {"decision": "allow", "pattern": "Shell(git status)"}, *_RULES
    ]


@needs_tomllib
def test_kimi_rules_already_in_place_leave_the_file_untouched(tmp_path):
    config = tmp_path / "config.toml"
    rules = KimiPermissionRules(config, "mnemo")
    rules.apply()
    before = config.read_bytes(), config.stat().st_mtime_ns

    assert rules.apply() == PRESENT
    assert (config.read_bytes(), config.stat().st_mtime_ns) == before


@needs_tomllib
def test_kimi_rules_add_only_the_missing_one(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        "[[permission.rules]]\n"
        'decision = "allow"\n'
        'pattern = "mcp__mnemo__*"\n',
        encoding="utf-8",
    )
    assert KimiPermissionRules(config, "mnemo").apply() == ADDED
    assert _rules_in(config) == _RULES


@needs_tomllib
@pytest.mark.parametrize("content", [
    'default_model = "kimi\n',  # not TOML
    "[permission]\nrules = []\n",  # an inline array a table cannot extend
    'permission = { dangerous_command_guard = false }\n',  # an inline table, the same
])
def test_kimi_rules_refuse_a_config_they_cannot_extend(tmp_path, content):
    config = tmp_path / "config.toml"
    config.write_text(content, encoding="utf-8")

    with pytest.raises(RulesNotWritten) as refused:
        KimiPermissionRules(config, "mnemo").apply()

    assert config.read_text(encoding="utf-8") == content
    assert refused.value.snippet == (
        "[[permission.rules]]\n"
        'decision = "allow"\n'
        'pattern = "mcp__mnemo__*"\n'
        "\n"
        "[[permission.rules]]\n"
        'decision = "ask"\n'
        'pattern = "mcp__mnemo__delete*"\n'
    )


def test_without_tomllib_kimi_rules_are_printed_not_written(tmp_path, monkeypatch):
    monkeypatch.setattr(kimi_permission_rules, "_toml_reader", lambda: None)
    config = tmp_path / "config.toml"

    with pytest.raises(RulesNotWritten) as refused:
        KimiPermissionRules(config, "mnemo").apply()

    assert "3.11" in refused.value.reason
    assert "mcp__mnemo__delete*" in refused.value.snippet
    assert not config.exists()


@needs_tomllib
def test_kimi_setup_wires_the_server_and_its_rules_in_kimi_home(tmp_path):
    kimi_home = tmp_path / "kimi-home"
    kimi = next(
        installer
        for installer in build_installers(
            connector_argv=["/opt/bin/mnemo-mcp"], runner=None, home=tmp_path,
            env={"KIMI_CODE_HOME": str(kimi_home)},
        )
        if installer.name == "kimi-code"
    )

    result = kimi.install()

    assert result.status == "ok" and result.message == ADDED
    assert json.loads((kimi_home / "mcp.json").read_text())["mcpServers"]["mnemo"] == {
        "command": "/opt/bin/mnemo-mcp", "args": []
    }
    assert _rules_in(kimi_home / "config.toml") == _RULES
