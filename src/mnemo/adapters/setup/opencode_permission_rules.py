"""OpenCode: mnemo's permission rules in opencode.json.

OpenCode names an MCP tool `<server>_<tool>` and applies the LAST rule that matches
it, so mnemo's rules go at the end, the ask rule for deletes after the allow rule for
every tool; rules of mnemo's already in the file are moved there.

OpenCode 1 reads `"permission": {..., "mnemo_*": "allow", "mnemo_delete*": "ask"}`,
in key order. OpenCode 2 reads that too and converts it in memory, so it stays the
default. `permission` may be a single word for every tool: "allow" already lets
mnemo's tools run, deletes included, so it is left as it is; "ask" or "deny" becomes
`{"*": word}`, which OpenCode reads the same way, followed by mnemo's keys.

OpenCode 2's own rules are a `permissions` list of `{action, resource, effect}`, and
it places them after the converted version-1 ones. So when the file has that list,
mnemo's rules go at its end instead; written into `permission`, a broader rule in the
list would override them. Every other key and setting is kept.
"""
from __future__ import annotations

import json
from pathlib import Path

from mnemo.adapters.setup.json_config import save_json
from mnemo.adapters.setup.permission_policy import POLICY
from mnemo.adapters.setup.permission_rules import (
    ADDED,
    PRESENT,
    RulesNotWritten,
    describe_rules,
    load_json_object,
)

ALREADY_ALLOWED = 'permission is "allow" for every tool; left as it is'


class OpencodePermissionRules:
    def __init__(self, config_path: Path, server_name: str) -> None:
        self._path = Path(config_path)
        self._rules = [(rule.decision, f"{server_name}_{rule.tools}") for rule in POLICY]
        self._keys = {pattern: decision for decision, pattern in self._rules}
        self._list = [
            {"action": pattern, "resource": "*", "effect": decision}
            for decision, pattern in self._rules
        ]
        self._snippet = json.dumps({"permission": self._keys}, indent=2)

    @property
    def target(self) -> Path:
        return self._path

    def describe(self) -> str:
        return describe_rules(self._rules, self._path)

    def apply(self) -> str:
        data = load_json_object(self._path, self._snippet)
        if "permissions" in data:
            return self._apply_to_list(data)

        permission = data.get("permission", {})
        if permission == "allow":
            return ALREADY_ALLOWED
        if permission in ("ask", "deny"):
            permission = {"*": permission}
        if not isinstance(permission, dict):
            raise RulesNotWritten(f"`permission` is {json.dumps(permission)}", self._snippet)

        keys = list(permission.items())
        if keys[-len(self._keys):] == list(self._keys.items()):
            return PRESENT
        kept = {key: value for key, value in keys if key not in self._keys}
        data["permission"] = {**kept, **self._keys}
        save_json(self._path, data)
        return ADDED

    def _apply_to_list(self, data: dict) -> str:
        rules = data["permissions"]
        if not isinstance(rules, list):
            snippet = json.dumps({"permissions": self._list}, indent=2)
            raise RulesNotWritten("`permissions` is not a list", snippet)
        if rules[-len(self._list):] == self._list:
            return PRESENT
        data["permissions"] = [rule for rule in rules if rule not in self._list] + self._list
        save_json(self._path, data)
        return ADDED
