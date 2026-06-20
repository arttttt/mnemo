"""Integration tests for the SQL executors against a real SQLite database."""
import pytest


def _executors(tmp_path):
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.executors import SqlReadExecutor, SqlWriteExecutor
    from mnemo.adapters.store.sqlite_connections import SqliteConnections

    conns = SqliteConnections(str(tmp_path / "x.db"))
    return conns, SqlReadExecutor(conns), SqlWriteExecutor(conns)


def test_write_commit_persists(tmp_path):
    conns, read, write = _executors(tmp_path)
    write.execute(lambda c: c.execute("CREATE TABLE t (x INTEGER)"))
    write.execute(lambda c: c.execute("INSERT INTO t VALUES (1)"))
    assert read.execute(lambda c: c.execute("SELECT count(*) FROM t").fetchone()[0]) == 1
    conns.close()


def test_write_rollback_discards_partial_unit(tmp_path):
    conns, read, write = _executors(tmp_path)
    write.execute(lambda c: c.execute("CREATE TABLE t (x INTEGER)"))

    def bad(c):
        c.execute("INSERT INTO t VALUES (1)")
        raise RuntimeError("boom after a write")

    with pytest.raises(RuntimeError):
        write.execute(bad)
    # the insert before the raise must NOT survive — the whole unit rolled back
    assert read.execute(lambda c: c.execute("SELECT count(*) FROM t").fetchone()[0]) == 0
    conns.close()


def test_write_returns_work_result(tmp_path):
    conns, _read, write = _executors(tmp_path)
    write.execute(lambda c: c.execute("CREATE TABLE t (x INTEGER)"))
    rowcount = write.execute(lambda c: c.execute("INSERT INTO t VALUES (1),(2)").rowcount)
    assert rowcount == 2
    conns.close()


def test_snapshot_read_runs_multiple_statements_and_returns(tmp_path):
    from mnemo.adapters.store.transaction import SnapshotRead

    conns, read, write = _executors(tmp_path)
    write.execute(lambda c: c.execute("CREATE TABLE t (x INTEGER)"))
    write.execute(lambda c: c.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)]))

    def two_legs(c):
        count = c.execute("SELECT count(*) FROM t").fetchone()[0]
        total = c.execute("SELECT sum(x) FROM t").fetchone()[0]
        return count, total

    assert read.execute(two_legs, strategy=SnapshotRead()) == (3, 6)
    # the snapshot transaction was closed, so the reader is usable again
    assert read.execute(lambda c: c.execute("SELECT 1").fetchone()[0]) == 1
    conns.close()
