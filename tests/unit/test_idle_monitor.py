"""IdleMonitor fires on_idle only after the grace elapses with no live connector.

Pure state-machine coverage with a scripted probe and tiny timings; the real
flock-backed liveness is exercised in tests/integration/test_idle_exit.py.
"""
import threading

from mnemo.adapters.mcp.idle_monitor import IdleMonitor


class _Probe:
    """Returns a scripted sequence of live-counts, then sticks on the last value."""

    def __init__(self, counts):
        self._counts = list(counts)
        self._i = 0

    def live_count(self) -> int:
        value = self._counts[min(self._i, len(self._counts) - 1)]
        self._i += 1
        return value


class _FlakyProbe:
    """Raises on the first live_count() call, then reports empty — to prove a probe error
    does not kill the monitor thread."""

    def __init__(self):
        self._calls = 0

    def live_count(self) -> int:
        self._calls += 1
        if self._calls == 1:
            raise OSError("probe boom")
        return 0


def _start(monitor: IdleMonitor) -> threading.Thread:
    thread = threading.Thread(target=monitor.run, daemon=True)
    thread.start()
    return thread


def test_exits_after_grace_when_empty():
    fired = threading.Event()
    monitor = IdleMonitor(_Probe([0]), on_idle=fired.set, grace_seconds=0.05, interval_seconds=0.01)
    thread = _start(monitor)
    assert fired.wait(2.0)
    thread.join(1.0)


def test_stays_while_a_connector_is_live():
    fired = threading.Event()
    monitor = IdleMonitor(_Probe([1]), on_idle=fired.set, grace_seconds=0.05, interval_seconds=0.01)
    thread = _start(monitor)
    assert not fired.wait(0.3)  # a live connector never idle-exits
    monitor.stop()
    thread.join(1.0)


def test_a_returning_connector_cancels_the_grace():
    fired = threading.Event()
    # Empty for two ticks (< grace), then a connector returns -> the timer resets.
    monitor = IdleMonitor(
        _Probe([0, 0, 1]), on_idle=fired.set, grace_seconds=0.1, interval_seconds=0.02
    )
    thread = _start(monitor)
    assert not fired.wait(0.4)
    monitor.stop()
    thread.join(1.0)


def test_monitor_survives_a_probe_error_and_still_idle_exits():
    fired = threading.Event()
    monitor = IdleMonitor(_FlakyProbe(), on_idle=fired.set, grace_seconds=0.05, interval_seconds=0.01)
    thread = _start(monitor)
    assert fired.wait(2.0)  # the first-tick probe error did not kill the monitor
    thread.join(1.0)
