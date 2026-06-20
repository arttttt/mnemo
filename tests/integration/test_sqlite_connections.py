"""SqliteConnections.close() releases the writer and every per-thread reader."""
import sqlite3

import pytest


def test_close_releases_writer_and_readers(tmp_path):
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.sqlite_connections import SqliteConnections

    conns = SqliteConnections(str(tmp_path / "x.db"))
    reader = conns.reader()
    assert reader.execute("SELECT 1").fetchone()[0] == 1  # open and usable

    conns.close()

    with pytest.raises(sqlite3.ProgrammingError):
        reader.execute("SELECT 1")  # the reader connection was closed
    with pytest.raises(sqlite3.ProgrammingError):
        with conns.writer() as writer:
            writer.execute("SELECT 1")  # the writer connection was closed


def test_connections_open_in_manual_mode(tmp_path):
    """A pure resource: every connection opens in manual mode (isolation_level is
    None) so the executors — not the connection manager — own every transaction.
    There is no implicit-transaction mode to fall back to."""
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.sqlite_connections import SqliteConnections

    conns = SqliteConnections(str(tmp_path / "x.db"))
    with conns.writer() as writer:
        assert writer.isolation_level is None
    assert conns.reader().isolation_level is None
    conns.close()
