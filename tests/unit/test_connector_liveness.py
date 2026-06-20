"""ConnectorLiveness counts live connectors and is the sole cleaner of markers.

The cross-process / crash cases (a held marker reads as live; a SIGKILLed
connector frees it) are covered at the real boundary in
tests/integration/test_idle_exit.py — flock liveness only has meaning across
processes. Here we pin the deterministic bits: an empty dir is zero, and a marker
with no holder (a dead connector) is counted as gone *and swept*.
"""
import errno
import os
from pathlib import Path

from mnemo.adapters.mcp.connector_liveness import ConnectorLiveness


def test_missing_dir_is_zero(tmp_path):
    assert ConnectorLiveness(tmp_path / "connectors").live_count() == 0


def test_empty_dir_is_zero(tmp_path):
    connectors = tmp_path / "connectors"
    connectors.mkdir()
    assert ConnectorLiveness(connectors).live_count() == 0


def test_stale_marker_is_not_live_and_gets_swept(tmp_path):
    connectors = tmp_path / "connectors"
    connectors.mkdir()
    stale = connectors / "dead-session.lock"
    stale.write_text("")  # a marker no process holds == a dead connector

    liveness = ConnectorLiveness(connectors)
    assert liveness.live_count() == 0
    assert not stale.exists()  # the probe is the sole cleaner — no leak


def test_marker_probe_error_assumes_live(tmp_path, monkeypatch):
    # An indeterminate OSError on a marker (here EMFILE from os.open) must read as LIVE and
    # never propagate — one bad marker can't blind the count or kill the monitor thread.
    connectors = tmp_path / "connectors"
    connectors.mkdir()
    (connectors / "agent.lock").write_text("")
    real_open = os.open

    def flaky_open(path, *args, **kwargs):
        if str(path).endswith(".lock"):
            raise OSError(errno.EMFILE, "too many open files")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", flaky_open)
    assert ConnectorLiveness(connectors).live_count() >= 1  # assumed live, did not raise


def test_scan_error_assumes_live(tmp_path, monkeypatch):
    # If the whole scan fails (e.g. EACCES on the dir), assume at least one live connector so
    # the service does not idle-exit on a scan it could not perform.
    connectors = tmp_path / "connectors"
    connectors.mkdir()

    def boom(*args, **kwargs):
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(Path, "glob", boom)
    assert ConnectorLiveness(connectors).live_count() == 1
