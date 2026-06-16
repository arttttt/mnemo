"""Stop the shared on-demand service — the inverse of ``launcher.ensure_service_running``.

An operation that changes what a *running* service holds in memory — a reindex that
rebuilds the store at a new embedding dimension, or a code update — must stop it, or the
stale process keeps serving the old state (and writing vectors of the wrong dimension).
There is no explicit re-start here: the connector brings the service back on demand with
the fresh state (the on-demand contract), so "restart" means "stop; let it respawn when
an agent next needs it".
"""
from __future__ import annotations

import os
import signal
import time
from pathlib import Path

from mnemo.adapters.mcp.run_paths import run_dir as _run_dir
from mnemo.infrastructure.config import Config


def stop_service(config: Config, *, timeout: float = 5.0) -> bool:
    """SIGTERM the running service and wait for it to exit; SIGKILL if it overstays.

    Returns ``True`` if a live service was stopped, ``False`` if none was running.
    Idempotent, and clears a stale pidfile either way.
    """
    pid_file = _run_dir(config) / "service.pid"
    pid = _read_pid(pid_file)
    if pid is None or not _is_alive(pid):
        pid_file.unlink(missing_ok=True)  # nothing running — drop a stale pidfile
        return False
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while _is_alive(pid):
        if time.monotonic() >= deadline:
            os.kill(pid, signal.SIGKILL)  # last resort
            break
        time.sleep(0.05)
    pid_file.unlink(missing_ok=True)
    return True


def _read_pid(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists, just not ours to signal
    return True
