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
