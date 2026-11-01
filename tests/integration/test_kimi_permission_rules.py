"""Kimi Code's permission rules, appended to config.toml as text.

The rest of the file must stay byte for byte, the rules must read back as the tables
Kimi Code parses, and a file that a new table cannot extend must be left alone.
Reading TOML needs tomllib (Python 3.11+); the tests that parse it skip on 3.10, where
setup wires the server but prints the rules instead of writing them, which is tested on
every version.
"""
import json
import sys

import pytest

from mnemo.adapters.setup import kimi_permission_rules
from mnemo.adapters.setup.client_registry import build_installers
from mnemo.adapters.setup.kimi_permission_rules import KimiPermissionRules
from mnemo.adapters.setup.permission_rules import ADDED, PRESENT, RulesNotWritten

needs_tomllib = pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib is 3.11+")

# Ask first: Kimi Code 2.4 checks ask rules before allow rules wherever they are, and
# its documentation says the first match wins; written in this order, deletes ask
# under either reading.
_RULES = [
    {"decision": "ask", "pattern": "mcp__mnemo__delete*"},
    {"decision": "allow", "pattern": "mcp__mnemo__*"},
]
_SNIPPET = (
    "[[permission.rules]]\n"
    'decision = "ask"\n'
    'pattern = "mcp__mnemo__delete*"\n'
    "\n"
    "[[permission.rules]]\n"
    'decision = "allow"\n'
    'pattern = "mcp__mnemo__*"\n'
)


def _rules_in(path):
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8"))["permission"]["rules"]


def _kimi(home, env):
    installers = build_installers(connector_argv=["/opt/bin/mnemo-mcp"], home=home, env=env)
    return next(installer for installer in installers if installer.name == "kimi-code")


@needs_tomllib
def test_kimi_rules_ask_before_deletes_and_allow_every_tool(tmp_path):
    config = tmp_path / ".kimi-code" / "config.toml"

    note = KimiPermissionRules(config, "mnemo").apply()

    assert note == ADDED
    assert config.read_text(encoding="utf-8") == _SNIPPET
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
def test_kimi_rules_keep_a_file_with_windows_line_endings_in_them(tmp_path):
    config = tmp_path / "config.toml"
    original = b'default_model = "kimi-k2"\r\n'
    config.write_bytes(original)

    KimiPermissionRules(config, "mnemo").apply()

    updated = config.read_bytes()
    assert updated.startswith(original)
    assert b"\n" not in updated.replace(b"\r\n", b"")
    assert _rules_in(config) == _RULES


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
    assert _rules_in(config) == [_RULES[1], _RULES[0]]


@needs_tomllib
@pytest.mark.parametrize("content", [
    'default_model = "kimi\n'.encode(),  # not TOML
    b"[permission]\nrules = []\n",  # an inline array a table cannot extend
    b"permission = { dangerous_command_guard = false }\n",  # an inline table, the same
    '# réglages\n'.encode("latin-1"),  # not UTF-8
])
def test_kimi_rules_refuse_a_config_they_cannot_extend(tmp_path, content):
    config = tmp_path / "config.toml"
    config.write_bytes(content)

    with pytest.raises(RulesNotWritten) as refused:
        KimiPermissionRules(config, "mnemo").apply()

    assert config.read_bytes() == content
    assert refused.value.snippet == _SNIPPET


def test_without_tomllib_kimi_rules_are_printed_not_written(tmp_path, monkeypatch):
    monkeypatch.setattr(kimi_permission_rules, "_toml_reader", lambda: None)
    rules = KimiPermissionRules(tmp_path / "config.toml", "mnemo")

    assert "by hand" in rules.describe() and "3.11" in rules.describe()
    with pytest.raises(RulesNotWritten) as refused:
        rules.apply()

    assert "3.11" in refused.value.reason
    assert refused.value.snippet == _SNIPPET
    assert not (tmp_path / "config.toml").exists()


def test_without_tomllib_kimi_setup_still_wires_the_server(tmp_path, monkeypatch):
    # Python 3.10: the server is wired; the rules are printed, and the client is reported
    # as failed so the user sees there is something left to add.
    monkeypatch.setattr(kimi_permission_rules, "_toml_reader", lambda: None)

    result = _kimi(tmp_path, env={}).install()

    assert result.status == "failed"
    assert _SNIPPET in result.message
    assert "mnemo" in json.loads((tmp_path / ".kimi-code" / "mcp.json").read_text())["mcpServers"]
    assert not (tmp_path / ".kimi-code" / "config.toml").exists()


@needs_tomllib
def test_kimi_setup_wires_the_server_and_its_rules_in_kimi_home(tmp_path):
    kimi_home = tmp_path / "kimi-home"

    result = _kimi(tmp_path, env={"KIMI_CODE_HOME": str(kimi_home)}).install()

    assert result.status == "ok" and result.message == ADDED
    assert json.loads((kimi_home / "mcp.json").read_text())["mcpServers"]["mnemo"] == {
        "command": "/opt/bin/mnemo-mcp", "args": []
    }
    assert _rules_in(kimi_home / "config.toml") == _RULES
