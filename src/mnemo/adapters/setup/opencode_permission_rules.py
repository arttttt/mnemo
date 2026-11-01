"""OpenCode: mnemo's permission rules in opencode.json.

`"permission": {..., "mnemo_*": "allow", "mnemo_delete*": "ask"}`. OpenCode names an
MCP tool `<server>_<tool>` and applies the LAST rule that matches it, in the order the
keys appear, so mnemo's keys go at the end, the ask rule for deletes after the allow
rule for every tool; keys of mnemo's already in the file are moved there.

`permission` may be a single word for every tool. "allow" already lets mnemo's tools
run, deletes included, so it is left as it is. "ask" or "deny" becomes `{"*": word}`,
which OpenCode reads the same way, followed by mnemo's keys. Every other key and
setting is kept.
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
        self._snippet = json.dumps({"permission": self._keys}, indent=2)

    @property
    def target(self) -> Path:
        return self._path

    def describe(self) -> str:
        return describe_rules(self._rules, self._path)

    def apply(self) -> str:
        data = load_json_object(self._path, self._snippet)
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
