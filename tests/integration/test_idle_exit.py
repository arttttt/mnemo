"""The shared service idle-exits once no connector is alive.

Drives the real boundary with short idle timings: a real `mnemo-service`
(and, for the crash case, a real `mnemo-mcp` connector) subprocess. Liveness is
keyed to the connector's flock, which the kernel frees on death — so these cover
the live-keeps-alive, clean-release, and SIGKILL (crash) paths end to end.
"""
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("uvicorn")
pytest.importorskip("sqlite_vec")

_SRC = Path(__file__).resolve().parents[2] / "src"
_GRACE = "1"
_INTERVAL = "0.2"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _is_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _env(host: str, port: int, data_dir: Path) -> dict:
    return {
        **os.environ,
        "PYTHONPATH": str(_SRC),
        "MNEMO_EMBEDDER": "hash",
        "MNEMO_DATA_DIR": str(data_dir),
        "MNEMO_HOST": host,
        "MNEMO_PORT": str(port),
        "MNEMO_IDLE_GRACE_SECONDS": _GRACE,
        "MNEMO_IDLE_CHECK_INTERVAL_SECONDS": _INTERVAL,
    }


def _wait(predicate, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def _spawn(entry: str, env: dict, **kwargs) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", f"from mnemo.adapters.mcp.{entry} import main; main()"],
        env=env,
        **kwargs,
    )


def test_exits_when_no_connector_ever_connects(tmp_path):
    """An orphan spawn that no connector uses exits after the grace period."""
    host, port = "127.0.0.1", _free_port()
    proc = _spawn("service", _env(host, port, tmp_path / "data"))
    try:
        assert _wait(lambda: proc.poll() is not None, timeout=8.0), "service did not idle-exit"
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_live_connector_keeps_service_then_exit_on_release(tmp_path):
    """A held marker keeps the service up past grace; releasing it lets it exit."""
    from mnemo.adapters.mcp.connector_presence import ConnectorPresence

    host, port = "127.0.0.1", _free_port()
    data_dir = tmp_path / "data"
    connectors = data_dir.parent / "run" / "connectors"

    # A live connector (this process holds the flock) is present before the service.
    presence = ConnectorPresence(connectors)
    presence.acquire("test-session")

    proc = _spawn("service", _env(host, port, data_dir))
    try:
        assert _wait(lambda: _is_listening(host, port), timeout=20.0), "service did not start"
        # Past the grace period, but still up: a connector is alive.
        time.sleep(float(_GRACE) + 1.0)
        assert proc.poll() is None and _is_listening(host, port)

        # The connector goes away cleanly -> the sweep reaps it -> the service exits.
        presence.release()
        assert _wait(lambda: proc.poll() is not None, timeout=6.0), "service did not exit after release"
        assert proc.returncode == 0
    finally:
        presence.release()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_sigkilled_connector_lets_service_exit(tmp_path):
    """The crash path: a SIGKILLed connector's flock is freed by the kernel, so
    the service it spawned idle-exits (no PID tracking, immune to PID reuse)."""
    host, port = "127.0.0.1", _free_port()
    data_dir = tmp_path / "data"

    # A real connector: it acquires presence, spawns the (detached) service, serves.
    connector = _spawn(
        "proxy",
        _env(host, port, data_dir),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait(lambda: _is_listening(host, port), timeout=20.0), "connector did not bring up service"

        # Kill the connector outright — no clean unregister runs.
        connector.send_signal(signal.SIGKILL)
        connector.wait(timeout=5)

        # The detached service survives the connector but now sees no live one -> exits.
        assert _wait(lambda: not _is_listening(host, port), timeout=8.0), "service did not idle-exit after crash"
    finally:
        if connector.poll() is None:
            connector.kill()
            connector.wait(timeout=5)
