"""ConnectorLiveness counts live connectors and is the sole cleaner of markers.

The cross-process / crash cases (a held marker reads as live; a SIGKILLed
connector frees it) are covered at the real boundary in
tests/integration/test_idle_exit.py — flock liveness only has meaning across
processes. Here we pin the deterministic bits: an empty dir is zero, and a marker
with no holder (a dead connector) is counted as gone *and swept*.
"""
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
