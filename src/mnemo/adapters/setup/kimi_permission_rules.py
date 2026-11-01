"""Kimi Code: mnemo's permission rules in config.toml.

Two `[[permission.rules]]` tables, ask for `mcp__mnemo__delete*` and allow for
`mcp__mnemo__*`, in config.toml in Kimi's home ($KIMI_CODE_HOME, else ~/.kimi-code).
Kimi Code 2.4 checks configured deny rules, then ask, then allow, whatever their order
in the file, so the ask rule for deletes wins over the allow rule. Its documentation
says instead that the first matching rule takes effect; the ask rule is written first,
so deletes ask under either reading.

The standard library reads TOML but does not write it, so missing rules are appended
to the file as text, in the file's own line endings, and the rest of the file,
comments included, stays byte for byte. The result is parsed before it is written:
when the file holds `permission` or `permission.rules` in a form a new table cannot
extend (an inline array, say), nothing is written. Reading TOML needs Python 3.11's
tomllib; on Python 3.10 nothing is written either. In both cases setup prints the
rules to add by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

from mnemo.adapters.setup.config_file import replace_file
from mnemo.adapters.setup.permission_policy import POLICY
from mnemo.adapters.setup.permission_rules import ADDED, PRESENT, RulesNotWritten, describe_rules

_NEEDS_TOMLLIB = "reading TOML needs Python 3.11 or later"


class KimiPermissionRules:
    def __init__(self, config_path: Path, server_name: str) -> None:
        self._path = Path(config_path)
        self._rules = sorted(
            ((rule.decision, f"mcp__{server_name}__{rule.tools}") for rule in POLICY),
            key=lambda rule: rule[0] != "ask",
        )
        self._snippet = _tables(self._rules, "\n")

    @property
    def target(self) -> Path:
        return self._path

    def describe(self) -> str:
        if _toml_reader() is None:
            return f"print permission rules to add to {self._path} by hand ({_NEEDS_TOMLLIB})"
        return describe_rules(self._rules, self._path)

    def apply(self) -> str:
        tomllib = _toml_reader()
        if tomllib is None:
            raise RulesNotWritten(_NEEDS_TOMLLIB, self._snippet)
        try:
            text = self._path.read_bytes().decode("utf-8") if self._path.exists() else ""
        except UnicodeDecodeError as error:
            raise RulesNotWritten(f"{self._path} is not UTF-8 text", self._snippet) from error
        try:
            configured = _configured(tomllib.loads(text))
        except tomllib.TOMLDecodeError as error:
            raise RulesNotWritten(f"{self._path} is not valid TOML ({error})", self._snippet) from error

        missing = [rule for rule in self._rules if rule not in configured]
        if not missing:
            return PRESENT
        newline = "\r\n" if "\r\n" in text else "\n"
        if text and not text.endswith("\n"):
            text += newline
        updated = text + (newline if text else "") + _tables(missing, newline)
        try:
            tomllib.loads(updated)
        except tomllib.TOMLDecodeError as error:
            raise RulesNotWritten(
                f"appending [[permission.rules]] to {self._path} would not be valid TOML ({error})",
                self._snippet,
            ) from error
        replace_file(self._path, updated.encode("utf-8"))
        return ADDED


def _toml_reader():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        return None
    return tomllib


def _configured(config: dict) -> set[tuple[str, str]]:
    """The (decision, pattern) of every rule already in `permission.rules`."""
    permission = config.get("permission")
    rules = permission.get("rules") if isinstance(permission, dict) else None
    if not isinstance(rules, list):
        return set()
    return {
        (rule.get("decision"), rule.get("pattern")) for rule in rules if isinstance(rule, dict)
    }


def _tables(rules: list[tuple[str, str]], newline: str) -> str:
    tables = "\n".join(
        f"[[permission.rules]]\ndecision = {json.dumps(decision)}\npattern = {json.dumps(pattern)}\n"
        for decision, pattern in rules
    )
    return tables.replace("\n", newline)
