"""Claude Code: mnemo's permission rules in the user settings file.

`{"permissions": {"allow": ["mcp__mnemo__*"], "ask": ["mcp__mnemo__delete*"]}}` in
~/.claude/settings.json ($CLAUDE_CONFIG_DIR/settings.json when that is set). Claude
Code checks deny rules, then ask, then allow, whatever their order in the file, so the
ask rule for deletes wins over the allow rule for every tool. Existing rules and every
other setting are kept.
"""
from __future__ import annotations

import json
from pathlib import Path

from mnemo.adapters.setup.json_config import load_json, save_json
from mnemo.adapters.setup.permission_policy import POLICY
from mnemo.adapters.setup.permission_rules import ADDED, PRESENT, RulesNotWritten, describe_rules


class ClaudePermissionRules:
    def __init__(self, settings_path: Path, server_name: str) -> None:
        self._path = Path(settings_path)
        self._rules = [
            (rule.decision, f"mcp__{server_name}__{rule.tools}") for rule in POLICY
        ]

    @property
    def target(self) -> Path:
        return self._path

    def describe(self) -> str:
        return describe_rules(self._rules, self._path)

    def apply(self) -> str:
        try:
            data = load_json(self._path)
        except ValueError as error:
            raise self._refuse(f"{self._path} is not valid JSON ({error})") from error
        if not isinstance(data, dict):
            raise self._refuse(f"{self._path} does not hold a JSON object")
        permissions = data.setdefault("permissions", {})
        if not isinstance(permissions, dict):
            raise self._refuse("`permissions` is not an object")

        added = False
        for decision, pattern in self._rules:
            entries = permissions.setdefault(decision, [])
            if not isinstance(entries, list):
                raise self._refuse(f"`permissions.{decision}` is not a list")
            if pattern not in entries:
                entries.append(pattern)
                added = True
        if not added:
            return PRESENT
        save_json(self._path, data)
        return ADDED

    def _refuse(self, reason: str) -> RulesNotWritten:
        grouped: dict[str, list[str]] = {}
        for decision, pattern in self._rules:
            grouped.setdefault(decision, []).append(pattern)
        return RulesNotWritten(reason, json.dumps({"permissions": grouped}, indent=2))
