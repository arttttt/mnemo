"""The server instructions stay short, and cite only names and values the tools accept.

Claude Code shows at most 4,096 characters of a server's instructions; the text keeps
well under that. A tool, parameter or value renamed without updating the text would tell
the agent to call something that does not exist, so everything the text cites in
backticks is checked against the pinned tool contract — the golden snapshot the wire
test keeps equal to what the service serves.
"""
import json
import re
from pathlib import Path

from mnemo.adapters.mcp.instructions import SERVER_INSTRUCTIONS

_GOLDEN_TOOLS = Path(__file__).parents[1] / "integration" / "golden" / "mcp_tools.json"
_BUDGET = 2048  # half of Claude Code's cut, to leave room as the text grows


def _contract():
    """Tool names, and each parameter's allowed values (empty when it is not an enum)."""
    tools = json.loads(_GOLDEN_TOOLS.read_text(encoding="utf-8"))
    parameters: dict[str, set] = {}
    for tool in tools.values():
        for name, schema in tool["inputSchema"].get("properties", {}).items():
            parameters.setdefault(name, set()).update(schema.get("enum", []))
    return set(tools), parameters


def _cited():
    """Each backticked span as (name, value); value is None for a bare name."""
    for span in re.findall(r"`([^`]+)`", SERVER_INSTRUCTIONS):
        name, _, value = span.partition("=")
        yield name, value.strip('"') if value else None


def test_instructions_stay_within_budget():
    assert len(SERVER_INSTRUCTIONS) <= _BUDGET


def test_every_cited_name_is_a_real_tool_or_parameter():
    tools, parameters = _contract()
    cited = {name for name, _ in _cited()}

    assert cited, "the text cites tools in backticks; without any, this test checks nothing"
    assert cited <= tools | set(parameters), f"unknown: {sorted(cited - tools - set(parameters))}"


def test_every_cited_value_is_one_the_parameter_accepts():
    _, parameters = _contract()
    values = [(name, value) for name, value in _cited() if value is not None]

    assert values, "the text cites a parameter value; without one, this test checks nothing"
    for name, value in values:
        assert value in parameters.get(name, set()), f"{name}={value!r}"
