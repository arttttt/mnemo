"""Shared fixtures for integration tests that drive the running shared service."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def free_tcp_port() -> int:
    return free_port()


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


def service_env(host: str, port: int, data_dir) -> dict:
    """Env for a `mnemo-service` / `mnemo-mcp` subprocess (offline backends)."""
    return {
        **os.environ,
        "PYTHONPATH": str(_SRC),
        "MNEMO_STORE": "memory",
        "MNEMO_EMBEDDER": "hash",
        "MNEMO_DATA_DIR": str(data_dir),
        "MNEMO_HOST": host,
        "MNEMO_PORT": str(port),
    }


@pytest.fixture
def service(tmp_path):
    """A running `mnemo-service` (offline: memory store + hash embedder).

    Yields ``(host, port)``.
    """
    pytest.importorskip("mcp")
    pytest.importorskip("uvicorn")
    host, port = "127.0.0.1", free_port()
    proc = subprocess.Popen(
        [sys.executable, "-c", "from mnemo.adapters.mcp.service import main; main()"],
        env=service_env(host, port, tmp_path),
    )
    try:
        _wait_until_listening(proc, host, port)
        yield host, port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
