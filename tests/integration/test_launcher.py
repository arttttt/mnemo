"""Readiness wait for an on-demand service spawn, and cleanup when it never comes up.

The lock-holder waits while the spawned process is alive and gives up if it dies or
overstays the cap (a hang). On giving up it must tear the spawn down — a lingering
orphan that binds later would make the next connector spawn a second service onto the
taken port and crash on bind().
"""
import pytest

from mnemo.adapters.mcp import launcher
from mnemo.adapters.mcp.run_paths import run_dir
from mnemo.infrastructure.config import Config


class _FakeProc:
    """Stands in for a Popen: poll() returns None while 'alive', else the exit code.
    Records kill() so the cleanup path can be asserted."""

    def __init__(self, returncode=None) -> None:
        self.returncode = returncode
        self.pid = 4242
        self.killed = False
        self.waited = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.waited = True
        return self.returncode


def _config(tmp_path, *, ready_timeout=0.2):
    return Config(
        data_dir=str(tmp_path / "data"),  # run_dir = tmp_path/run (off the real ~/.mnemo)
        embedder="hash",
        service_ready_timeout=ready_timeout,
    )


# --- _wait_until_listening ---


def test_wait_returns_once_the_service_listens(monkeypatch):
    monkeypatch.setattr(launcher, "_is_listening", lambda host, port: True)
    launcher._wait_until_listening(_FakeProc(), "127.0.0.1", 1, timeout=5.0)  # no raise


def test_wait_fails_fast_when_the_process_dies(monkeypatch):
    # Never listening + a process that has exited → fail fast with the exit code, rather
    # than waiting out the (here large) timeout and never releasing the lock.
    monkeypatch.setattr(launcher, "_is_listening", lambda host, port: False)
    with pytest.raises(RuntimeError, match="exited with code 1"):
        launcher._wait_until_listening(_FakeProc(returncode=1), "127.0.0.1", 1, timeout=600.0)


def test_wait_times_out_if_alive_but_never_listening(monkeypatch):
    # Alive (poll None) but never accepting → bounded by the configurable timeout.
    monkeypatch.setattr(launcher, "_is_listening", lambda host, port: False)
    with pytest.raises(TimeoutError):
        launcher._wait_until_listening(_FakeProc(returncode=None), "127.0.0.1", 1, timeout=0.2)


# --- _terminate ---


def test_terminate_kills_a_live_spawn():
    proc = _FakeProc(returncode=None)  # still alive
    launcher._terminate(proc)
    # wait() must be called: it reaps the process so the lock isn't released mid-kill.
    assert proc.killed and proc.waited and proc.poll() is not None


def test_terminate_is_a_noop_on_an_already_dead_spawn():
    proc = _FakeProc(returncode=0)  # already exited
    launcher._terminate(proc)
    assert not proc.killed  # no signal sent to a dead process


# --- ensure_service_running cleanup on a failed spawn ---


def test_a_timed_out_spawn_is_killed_and_its_pidfile_cleared(tmp_path, monkeypatch):
    config = _config(tmp_path)
    proc = _FakeProc(returncode=None)  # alive but never binds
    monkeypatch.setattr(launcher, "_is_listening", lambda host, port: False)
    monkeypatch.setattr(launcher, "_spawn_service", lambda run: proc)

    with pytest.raises(TimeoutError):
        launcher.ensure_service_running(config)

    assert proc.killed  # orphan torn down so the next connector won't double-spawn
    assert not (run_dir(config) / "service.pid").exists()  # stale pidfile dropped


def test_a_crashed_spawn_clears_its_pidfile_without_signalling(tmp_path, monkeypatch):
    config = _config(tmp_path, ready_timeout=600.0)  # large: the proc dies, doesn't time out
    proc = _FakeProc(returncode=1)  # already exited before listening
    monkeypatch.setattr(launcher, "_is_listening", lambda host, port: False)
    monkeypatch.setattr(launcher, "_spawn_service", lambda run: proc)

    with pytest.raises(RuntimeError, match="exited with code 1"):
        launcher.ensure_service_running(config)

    assert not proc.killed  # already dead — nothing to signal
    assert not (run_dir(config) / "service.pid").exists()
