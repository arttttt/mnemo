"""Bring the shared service up on demand (used by the per-agent connector).

There is no resident daemon, so the connector starts the service itself when it
is not already running. A file lock makes a burst of connectors spawn exactly
one service: the winner holds the lock while it waits for the service to accept
connections, so latecomers take the lock only after it is already up and skip
the spawn. The service is detached, so it outlives the connector and shuts
itself down on idle (a later step). A pidfile records the process.
"""
from __future__ import annotations

import fcntl
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from mnemo.adapters.mcp.run_paths import run_dir as _run_dir
from mnemo.infrastructure.config import Config

_READY_TIMEOUT = 30.0


def ensure_service_running(config: Config) -> None:
    if _is_listening(config.host, config.port):
        return
    run_dir = _run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "service.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _is_listening(config.host, config.port):
            return  # another connector brought it up while we waited for the lock
        pid_file = run_dir / "service.pid"
        proc = _spawn_service(run_dir)
        pid_file.write_text(str(proc.pid))
        # Hold the lock until the service binds the port, so a burst spawns exactly one
        # (latecomers take the lock only once it is already listening, and skip the spawn).
        # With the default ONNX embedder the model loads lazily in a worker after the port
        # is up, so binding is fast; the generous, configurable cap covers the slower cases
        # (an embedder that loads eagerly at startup, a cold download) and guards against a
        # process that comes up alive but never binds (a hang).
        try:
            _wait_until_listening(
                proc, config.host, config.port, config.service_ready_timeout
            )
        except BaseException:
            # Could not confirm the spawn bound the port (it timed out still alive, or it
            # died). Tear it down while we still hold the lock and drop its now-stale
            # pidfile: a lingering orphan that binds later would make the next connector —
            # which sees nothing listening, takes the freed lock, and spawns again — race a
            # second service onto the port and crash on bind().
            _terminate(proc)
            pid_file.unlink(missing_ok=True)
            raise


def _is_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _spawn_service(run_dir: Path) -> subprocess.Popen:
    log = open(run_dir / "service.log", "a")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", "from mnemo.adapters.mcp.service import main; main()"],
            env=os.environ.copy(),  # the service inherits the same MNEMO_* config
            stdout=log,
            stderr=log,
            start_new_session=True,  # detach: the service outlives this connector
        )
    finally:
        log.close()  # the child keeps its own dup of the fd
    return proc


def _wait_until_listening(
    proc: subprocess.Popen, host: str, port: int, timeout: float = _READY_TIMEOUT
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_listening(host, port):
            return
        if proc.poll() is not None:
            # The service died before it ever listened — fail fast (and release the lock)
            # so the next connector can respawn, rather than waiting out the whole timeout.
            raise RuntimeError(
                f"mnemo service exited with code {proc.returncode} before listening on "
                f"{host}:{port}; see service.log"
            )
        time.sleep(0.1)
    raise TimeoutError(f"service did not become ready on {host}:{port} within {timeout:.0f}s")


def _terminate(proc: subprocess.Popen) -> None:
    """Force-kill a spawned service we could not confirm as ready. It never bound the port
    or served anything, so there is nothing to shut down gracefully — SIGKILL guarantees it
    dies (even if it hung) so the port frees for the next attempt. No-op if it already exited."""
    if proc.poll() is not None:
        return  # already gone (e.g. it crashed before listening)
    proc.kill()
    proc.wait()
