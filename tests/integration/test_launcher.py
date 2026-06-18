"""Readiness wait for an on-demand service spawn.

The lock-holder must keep waiting while the spawned process is alive (a cold model
load is slow) and only give up if it dies — releasing the lock mid-load would let the
next connector spawn a second service onto the taken port.
"""
import pytest

from mnemo.adapters.mcp import launcher


class _FakeProc:
    """Stands in for a Popen: poll() returns None while 'alive', else the exit code."""

    def __init__(self, returncode=None) -> None:
        self.returncode = returncode

    def poll(self):
        return self.returncode


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
