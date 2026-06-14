"""A connector's liveness marker: a flock the kernel releases when it dies.

Each connector run holds an exclusive ``flock`` on
``<run>/connectors/<session_id>.lock`` for its whole life. The kernel releases
the lock automatically when the process exits for *any* reason — clean exit or
SIGKILL — so the service can tell a live connector from a dead one without
tracking PIDs (and is therefore immune to PID reuse). The connector never
deletes the file; the service's sweep is the sole cleaner.
"""
from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path

_ACQUIRE_RETRIES = 50
_ACQUIRE_BACKOFF = 0.01


class ConnectorPresence:
    def __init__(self, connectors_dir: Path) -> None:
        self._dir = Path(connectors_dir)
        self._fd: int | None = None

    def acquire(self, session_id: str) -> None:
        """Create and hold the marker for ``session_id``; keep the fd for the run."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{session_id}.lock"
        for _ in range(_ACQUIRE_RETRIES):
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # The sweep holds the lock for the instant it checks/unlinks; retry.
                os.close(fd)
                time.sleep(_ACQUIRE_BACKOFF)
                continue
            if os.fstat(fd).st_nlink == 0:
                # A sweep unlinked it in the open->lock window; recreate and retry.
                os.close(fd)
                continue
            self._fd = fd
            return
        raise RuntimeError(f"could not acquire presence marker for {session_id}")

    def release(self) -> None:
        """Release the marker. Production never calls this (process death frees the
        lock); tests use it to simulate a connector going away."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
