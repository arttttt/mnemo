"""The permission rules setup writes agree with the tools' own side-effect hints.

Codex asks before exactly the tools annotated destructive; the rules written for the
other clients must ask before the same tools, or a delete would run unasked in one
client and not another. A destructive tool named outside the `delete*` glob fails here.
Checked against the pinned tool contract, the golden the wire test keeps equal to what
the service serves.
"""
import json
from fnmatch import fnmatchcase
from pathlib import Path

from mnemo.adapters.setup.permission_policy import POLICY

_GOLDEN_TOOLS = Path(__file__).parents[1] / "integration" / "golden" / "mcp_tools.json"


def _tools():
    return json.loads(_GOLDEN_TOOLS.read_text(encoding="utf-8"))


def _covered(decision):
    globs = [rule.tools for rule in POLICY if rule.decision == decision]
    return {name for name in _tools() if any(fnmatchcase(name, glob) for glob in globs)}


def test_every_tool_is_allowed():
    assert _covered("allow") == set(_tools())


def test_the_ask_rule_covers_exactly_the_destructive_tools():
    destructive = {
        name for name, tool in _tools().items() if tool["annotations"].get("destructiveHint")
    }

    assert destructive, "the contract has destructive tools; without any, this checks nothing"
    assert _covered("ask") == destructive
