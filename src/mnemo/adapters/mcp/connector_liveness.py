"""Service-side probe: counts live connectors and cleans up dead markers.

For each connector marker it *tries* to take the flock: a lock it can take means
the owner is gone (the kernel freed it), so the marker is stale and gets removed;
a lock held by someone is a live connector. This is the sole cleaner of the
marker files — a marker lives only as long as its connector, plus at most one
sweep interval after the connector dies.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path


class ConnectorLiveness:
    def __init__(self, connectors_dir: Path) -> None:
        self._dir = Path(connectors_dir)

    def live_count(self) -> int:
        if not self._dir.exists():
            return 0
        return sum(1 for path in self._dir.glob("*.lock") if self._owner_alive(path))

    def _owner_alive(self, path: Path) -> bool:
        try:
            fd = os.open(path, os.O_RDWR)
        except FileNotFoundError:
            return False
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True  # held by a live connector
            # We took the lock -> the owner is gone -> remove the stale marker.
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            return False
        finally:
            os.close(fd)
