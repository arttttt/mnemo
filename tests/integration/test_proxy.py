"""The `mnemo-mcp` proxy forwards agent calls to the shared service.

Exercises BOTH components together along the full agent path: a stdio MCP client
-> the `mnemo-mcp` proxy (subprocess) -> the `mnemo-service` (subprocess) -> the
store, and back.
"""
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("uvicorn")
pytest.importorskip("sqlite_vec")

_SRC = Path(__file__).resolve().parents[2] / "src"


def _text(result) -> str:
    return " ".join(getattr(block, "text", "") or "" for block in result.content)


def _proxy_params(host: str, port: int, data_dir):
    from mcp.client.stdio import StdioServerParameters

    return StdioServerParameters(
        command=sys.executable,
        args=["-c", "from mnemo.adapters.mcp.proxy import main; main()"],
        env={
            **os.environ,
            "PYTHONPATH": str(_SRC),
            "MNEMO_DATA_DIR": str(data_dir),
            "MNEMO_HOST": host,
            "MNEMO_PORT": str(port),
        },
    )


def test_proxy_forwards_agent_calls_to_the_service(service, tmp_path):
    import anyio

    host, port = service

    async def flow():
        from mcp.client.session import ClientSession
        from mcp.client.stdio import stdio_client

        async with stdio_client(_proxy_params(host, port, tmp_path)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # The proxy exposes the service's tools (it knows nothing itself).
                tools = {tool.name for tool in (await session.list_tools()).tools}
                assert {"remember", "search"} <= tools

                await session.call_tool("create_project", {"name": "api"})
                stored = await session.call_tool(
                    "remember", {"content": "redis via the proxy", "project": "api"}
                )
                assert not stored.isError, _text(stored)

                hits = await session.call_tool("search", {"query": "redis", "scope": "all"})
                assert "redis via the proxy" in _text(hits)

    anyio.run(flow)


async def _proxy_create_project(host: str, port: int, data_dir, slug: str) -> None:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(_proxy_params(host, port, data_dir)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool("create_project", {"name": slug})


async def _proxy_writes(host: str, port: int, data_dir, contents: list[str]) -> None:
    """One proxy run (one agent) that writes several memories through it."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(_proxy_params(host, port, data_dir)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for content in contents:
                await session.call_tool("remember", {"content": content, "project": "api"})


def test_each_proxy_run_stamps_its_own_session_id(service, tmp_path):
    import anyio

    from mnemo.adapters.embedding.hash_embedder import HashEmbedder
    from mnemo.adapters.store.sqlite_vec_repository import SqliteRepositoryImpl

    host, port = service

    anyio.run(_proxy_create_project, host, port, tmp_path, "api")  # register once for both runs
    # One proxy run (one agent) writes twice → those memories share its session id.
    anyio.run(_proxy_writes, host, port, tmp_path, ["a1 via proxy one", "a2 via proxy one"])
    # A second, separate proxy run → its own session id.
    anyio.run(_proxy_writes, host, port, tmp_path, ["b1 via proxy two"])

    repo = SqliteRepositoryImpl.open(path=str(tmp_path / "memory.db"), dim=HashEmbedder().dim)
    session_id = {m.content: m.session_id for m in repo.list_all()}
    assert session_id["a1 via proxy one"] == session_id["a2 via proxy one"]  # one run = one id
    assert session_id["b1 via proxy two"] != session_id["a1 via proxy one"]  # other agent = other id
    assert all(session_id.values())  # the service stamped the proxy-supplied id (not null)


def test_proxy_spawns_the_service_when_down(free_tcp_port, tmp_path):
    """No service is running: the proxy must bring one up and then serve calls."""
    import signal

    import anyio

    host, port = "127.0.0.1", free_tcp_port
    data_dir = tmp_path / "data"
    pidfile = data_dir.parent / "run" / "service.pid"  # ~/.mnemo/run mirror

    env = {
        **os.environ,
        "PYTHONPATH": str(_SRC),
        "MNEMO_EMBEDDER": "hash",
        "MNEMO_DATA_DIR": str(data_dir),
        "MNEMO_HOST": host,
        "MNEMO_PORT": str(port),
    }

    async def flow():
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=["-c", "from mnemo.adapters.mcp.proxy import main; main()"],
            env=env,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool("create_project", {"name": "api"})
                stored = await session.call_tool(
                    "remember", {"content": "spawned by the proxy", "project": "api"}
                )
                assert not stored.isError, _text(stored)
                hits = await session.call_tool("search", {"query": "spawned", "scope": "all"})
                assert "spawned by the proxy" in _text(hits)

    try:
        anyio.run(flow)
    finally:
        # The spawned service is detached (no idle-exit yet), so stop it explicitly.
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass


def test_proxy_reconnects_after_the_service_is_killed(free_tcp_port, tmp_path):
    """Mid-session the service dies (the `mnemo reindex` scenario kills it). The next call
    must respawn it and succeed, instead of leaving the connector wedged on a dead stream."""
    import signal
    import socket
    import time

    import anyio

    host, port = "127.0.0.1", free_tcp_port
    data_dir = tmp_path / "data"
    pidfile = data_dir.parent / "run" / "service.pid"
    env = {
        **os.environ,
        "PYTHONPATH": str(_SRC),
        "MNEMO_EMBEDDER": "hash",
        "MNEMO_DATA_DIR": str(data_dir),
        "MNEMO_HOST": host,
        "MNEMO_PORT": str(port),
    }

    def _kill_service():
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass

    def _wait_port_free(timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=0.25):
                    pass
            except OSError:
                return
            time.sleep(0.05)

    async def flow():
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=["-c", "from mnemo.adapters.mcp.proxy import main; main()"],
            env=env,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                # Registered before the kill; it must survive the respawn (persisted),
                # just as memories do — else the post-respawn write would hit the gate.
                await session.call_tool("create_project", {"name": "api"})
                first = await session.call_tool(
                    "remember", {"content": "before the kill", "project": "api"}
                )
                assert not first.isError, _text(first)

                # Kill the service out from under the live connector, mid-session.
                await anyio.to_thread.run_sync(_kill_service)
                await anyio.to_thread.run_sync(_wait_port_free)

                # The next call must respawn the service and go through.
                after = await session.call_tool(
                    "remember", {"content": "after the respawn", "project": "api"}
                )
                assert not after.isError, _text(after)
                hits = await session.call_tool("search", {"query": "respawn", "scope": "all"})
                assert "after the respawn" in _text(hits)

    try:
        anyio.run(flow)
    finally:
        _kill_service()
