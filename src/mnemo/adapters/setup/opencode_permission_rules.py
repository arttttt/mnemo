"""OpenCode: mnemo's permission rules in opencode.json.

OpenCode names an MCP tool `<server>_<tool>` and applies the LAST rule that matches
it, so mnemo's rules go at the end, the ask rule for deletes after the allow rule for
every tool; mnemo's own rules already in the file are moved there. A rule the user set
for one of the same keys, with another value ("deny", say), is theirs: the file is
left as it is and setup says so, rather than lift it.

What the rules change differs by version. OpenCode 2 asks before every MCP call its
config does not allow, so both rules matter there. OpenCode 1 allows every tool by
default in its main agents (build, plan, general), so there only the ask rule for
deletes changes anything. Both apply the config's rules to every agent, after the
agent's own defaults: agents that deny every tool by default, such as explore, can then
use mnemo's tools too.

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
        permission = data.get("permission", {})
        # Checked before choosing where the rules go: in a file that also has OpenCode 2's
        # `permissions` list, the rules added to that list would override these.
        if isinstance(permission, dict):
            own = [
                f"{key}: {json.dumps(permission[key])}"
                for key, value in self._keys.items()
                if key in permission and permission[key] != value
            ]
            if own:
                raise RulesNotWritten(_users_own(own), self._snippet)
        if "permissions" in data:
            return self._apply_to_list(data)

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
        snippet = json.dumps({"permissions": self._list}, indent=2)
        if not isinstance(rules, list):
            raise RulesNotWritten("`permissions` is not a list", snippet)
        actions = set(self._keys)
        own = [
            json.dumps(rule)
            for rule in rules
            if isinstance(rule, dict) and rule.get("action") in actions and rule not in self._list
        ]
        if own:
            raise RulesNotWritten(_users_own(own), snippet)

        if rules[-len(self._list):] == self._list:
            return PRESENT
        data["permissions"] = [rule for rule in rules if rule not in self._list] + self._list
        save_json(self._path, data)
        return ADDED


def _users_own(rules: list[str]) -> str:
    return f"the file already has its own rules for mnemo's tools ({'; '.join(rules)}), left as they are"
