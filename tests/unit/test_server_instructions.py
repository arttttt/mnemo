"""The server instructions fit what every client shows whole, and cite only real names.

Claude Code cuts instructions at 4,096 characters and its older versions at 2,048; the
text has to survive the stricter cut. A tool or parameter renamed without updating the
text would tell the agent to call something that does not exist, so every name the text
cites in backticks is checked against the pinned tool contract — the golden snapshot the
wire test keeps equal to what the service serves.
"""
import json
import re
from pathlib import Path

from mnemo.adapters.mcp.instructions import SERVER_INSTRUCTIONS

_GOLDEN_TOOLS = Path(__file__).parents[1] / "integration" / "golden" / "mcp_tools.json"
_STRICTEST_CLIENT_LIMIT = 2048


def test_instructions_fit_the_strictest_client_limit():
    assert len(SERVER_INSTRUCTIONS) <= _STRICTEST_CLIENT_LIMIT


def test_every_name_the_instructions_cite_is_a_real_tool_or_parameter():
    tools = json.loads(_GOLDEN_TOOLS.read_text(encoding="utf-8"))
    known = set(tools)
    for tool in tools.values():
        known |= set(tool["inputSchema"].get("properties", {}))

    # `scope="global"` cites the parameter `scope`.
    cited = {span.split("=")[0] for span in re.findall(r"`([^`]+)`", SERVER_INSTRUCTIONS)}

    assert cited, "the text cites tools in backticks; without any, this test checks nothing"
    assert cited <= known, f"not a tool or parameter: {sorted(cited - known)}"
