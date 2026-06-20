"""Idle-exit monitor: shut the service down once no connector is alive.

Runs as a daemon thread. Every ``interval`` seconds it asks the liveness probe
how many connectors are alive. While some are, it does nothing. When none are it
starts a grace period; a connector appearing within it resets the timer; if the
grace elapses with still none, it fires ``on_idle`` (which stops the service).

The clock starts at boot treated as "the last one just left", so an orphan spawn
that no connector ever uses also exits after the grace period.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from mnemo.adapters.mcp.liveness_probe import LivenessProbe

_log = logging.getLogger("mnemo.idle")


class IdleMonitor:
    def __init__(
        self,
        liveness: LivenessProbe,
        on_idle: Callable[[], None],
        grace_seconds: float,
        interval_seconds: float,
    ) -> None:
        self._liveness = liveness
        self._on_idle = on_idle
        self._grace = grace_seconds
        self._interval = interval_seconds
        self._stop = threading.Event()

    def run(self) -> None:
        empty_since: float | None = time.monotonic()  # boot == "last one left"
        while not self._stop.wait(self._interval):
            try:
                live = self._liveness.live_count()
            except Exception:  # the probe self-handles, but never let it kill the monitor
                _log.warning(
                    "idle monitor: liveness probe raised; assuming live this tick", exc_info=True
                )
                empty_since = None  # assume live -> don't advance toward idle-exit
                continue
            if live > 0:
                empty_since = None
                continue
            if empty_since is None:
                empty_since = time.monotonic()
            elif time.monotonic() - empty_since >= self._grace:
                self._fire_idle()
                return

    def _fire_idle(self) -> None:
        # The grace elapsed with a trustworthy zero -> we have DECIDED to exit. A failure in
        # the shutdown callback must not abort that into a silent hang; log and return so the
        # process still tears down (the callback drives os._exit).
        try:
            self._on_idle()
        except Exception:
            _log.error("idle monitor: shutdown callback failed", exc_info=True)

    def stop(self) -> None:
        self._stop.set()
