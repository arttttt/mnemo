"""The per-agent proxy stays light to import.

One proxy runs per agent, and the footprint goal is a thin connector beside one shared
service. The proxy loads the MCP SDK only inside `_serve`, when it starts serving; what
it imports at module level (the server instructions among them) must not pull in the
SDK, the embedder stack or the store. Checked in a fresh interpreter, since this test
process has long since imported all of them.
"""
import os
import subprocess
import sys

from tests.support.mcp_wire import SRC

_HEAVY = ("mcp", "pydantic", "anyio", "httpx", "numpy", "onnxruntime", "sqlite_vec")


def test_importing_the_proxy_loads_nothing_heavy():
    probe = (
        "import sys\n"
        "import mnemo.adapters.mcp.proxy\n"
        f"print(','.join(m for m in {_HEAVY!r} if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )

    assert result.stdout.strip() == "", f"the proxy import loaded: {result.stdout.strip()}"
