"""Service-side probe: counts live connectors and cleans up dead markers.

For each connector marker it *tries* to take the flock: a lock it can take means
the owner is gone (the kernel freed it), so the marker is stale and gets removed;
a lock held by someone is a live connector. This is the sole cleaner of the
marker files — a marker lives only as long as its connector, plus at most one
sweep interval after the connector dies.
"""
from __future__ import annotations

import fcntl
import logging
import os
from pathlib import Path

_log = logging.getLogger("mnemo.liveness")


class ConnectorLiveness:
    def __init__(self, connectors_dir: Path) -> None:
        self._dir = Path(connectors_dir)
        self._last_fault: str | None = None  # signature of the last logged probe fault
        self._fault_this_scan = False

    def live_count(self) -> int:
        self._fault_this_scan = False
        try:
            if not self._dir.exists():
                count = 0
            else:
                count = sum(1 for path in self._dir.glob("*.lock") if self._owner_alive(path))
        except OSError as error:
            # The whole scan failed (e.g. EACCES/EMFILE on the dir). Assume at least one live
            # connector so the service does NOT idle-exit on a scan it could not perform.
            self._note_fault(f"scan:{error.errno}", error)
            return 1
        if not self._fault_this_scan:
            self._clear_fault()
        return count

    def _owner_alive(self, path: Path) -> bool:
        try:
            fd = os.open(path, os.O_RDWR)
        except FileNotFoundError:
            return False  # the marker is provably gone
        except OSError as error:
            # Indeterminate (EACCES/EMFILE/EIO/...): assume LIVE so we don't idle-exit while a
            # connector may still hold it — staying up is the recoverable direction.
            self._note_fault(f"open:{error.errno}", error)
            return True
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True  # held by a live connector
            except OSError as error:
                self._note_fault(f"flock:{error.errno}", error)
                return True  # can't determine -> assume live
            # We took the lock -> the owner is gone -> remove the stale marker.
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError as error:
                # Owner is provably gone (we hold the lock) -> not live; we just couldn't sweep
                # the marker. Swallow so an un-removable dead marker doesn't re-kill every scan.
                self._note_fault(f"unlink:{error.errno}", error)
            return False
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def _note_fault(self, signature: str, error: OSError) -> None:
        self._fault_this_scan = True
        if signature != self._last_fault:  # edge-triggered: log on a new/changed fault only
            _log.warning(
                "connector liveness probe error (%s); assuming live", error, exc_info=True
            )
            self._last_fault = signature

    def _clear_fault(self) -> None:
        if self._last_fault is not None:
            _log.info("connector liveness probe recovered")
            self._last_fault = None
