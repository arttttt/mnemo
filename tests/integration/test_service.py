"""The shared service serves MCP over streamable-http, and several agents share it.

Spins up `mnemo-service` in a subprocess (offline: memory store + hash embedder)
and drives it through the real MCP streamable-http client — the actual boundary.
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("uvicorn")

_SRC = Path(__file__).resolve().parents[2] / "src"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_until_listening(proc, host: str, port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"service exited early with code {proc.returncode}")
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"service did not start on {host}:{port}")


@pytest.fixture
def service_url(tmp_path):
    host, port = "127.0.0.1", _free_port()
    env = {
        **os.environ,
        "PYTHONPATH": str(_SRC),
        "MNEMO_STORE": "memory",
        "MNEMO_EMBEDDER": "hash",
        "MNEMO_DATA_DIR": str(tmp_path),
        "MNEMO_HOST": host,
        "MNEMO_PORT": str(port),
    }
    proc = subprocess.Popen(
        [sys.executable, "-c", "from mnemo.adapters.mcp.service import main; main()"],
        env=env,
    )
    try:
        _wait_until_listening(proc, host, port)
        yield f"http://{host}:{port}/mcp"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


async def _call(url: str, tool: str, args: dict):
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool, args)


def _text(result) -> str:
    return " ".join(getattr(block, "text", "") or "" for block in result.content)


def test_service_serves_tools_over_http(service_url):
    import anyio

    stored = anyio.run(
        _call, service_url, "remember", {"content": "redis caching layer", "project": "api"}
    )
    assert not stored.isError, _text(stored)

    hits = anyio.run(_call, service_url, "search", {"query": "redis cache", "scope": "all"})
    assert not hits.isError
    assert "redis caching layer" in _text(hits)


def test_several_connections_share_one_service(service_url):
    import anyio

    # One connection writes; a SEPARATE connection reads it back → both are served
    # by the one process (the single shared embedder + store behind them).
    anyio.run(
        _call, service_url, "remember", {"content": "jwt rotation policy", "project": "api"}
    )
    hits = anyio.run(_call, service_url, "search", {"query": "jwt rotation", "scope": "all"})
    assert "jwt rotation policy" in _text(hits)
