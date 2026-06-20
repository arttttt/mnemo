"""Unit tests for the SQL executors and transaction strategies.

These drive the executor's orchestration (begin/commit/rollback, busy-retry,
read snapshots) over a FAKE connection, so no real database is needed.
"""
import sqlite3
from contextlib import contextmanager

import pytest

from mnemo.adapters.store.executors import SqlReadExecutor, SqlWriteExecutor
from mnemo.adapters.store.transaction import (
    BusyRetry,
    Deferred,
    NoRetry,
    PlainRead,
    SnapshotRead,
)


class FakeConn:
    """Records executed SQL and tracks transaction state by the verbs it sees."""

    def __init__(self):
        self.calls: list[str] = []
        self._in_txn = False

    def execute(self, sql, *params):
        self.calls.append(sql)
        verb = sql.strip().upper()
        if verb.startswith("BEGIN"):
            self._in_txn = True
        elif verb.startswith(("COMMIT", "ROLLBACK")):
            self._in_txn = False
        return self

    @property
    def in_transaction(self) -> bool:
        return self._in_txn


class FakeConns:
    def __init__(self, conn: FakeConn):
        self._conn = conn

    @contextmanager
    def writer(self):
        yield self._conn

    def reader(self):
        return self._conn


# --- write executor ---


def test_write_commits_and_returns_result():
    conn = FakeConn()
    result = SqlWriteExecutor(FakeConns(conn)).execute(lambda c: "ok")
    assert result == "ok"
    assert conn.calls == ["BEGIN IMMEDIATE", "COMMIT"]


def test_write_rolls_back_on_error():
    conn = FakeConn()

    def work(c):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        SqlWriteExecutor(FakeConns(conn), retry=NoRetry()).execute(work)
    assert conn.calls == ["BEGIN IMMEDIATE", "ROLLBACK"]


def test_write_strategy_override_uses_deferred_begin():
    conn = FakeConn()
    SqlWriteExecutor(FakeConns(conn)).execute(lambda c: None, strategy=Deferred())
    assert conn.calls[0] == "BEGIN"


def test_write_retries_busy_then_succeeds():
    conn = FakeConn()
    seen = {"n": 0}

    def work(c):
        seen["n"] += 1
        if seen["n"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return "done"

    executor = SqlWriteExecutor(FakeConns(conn), retry=BusyRetry(max_attempts=5, base_delay=0))
    assert executor.execute(work) == "done"
    assert seen["n"] == 3
    # each failed attempt rolled back before the final commit
    assert conn.calls.count("ROLLBACK") == 2
    assert conn.calls[-1] == "COMMIT"


def test_write_does_not_retry_non_busy_operational_error():
    conn = FakeConn()
    seen = {"n": 0}

    def work(c):
        seen["n"] += 1
        raise sqlite3.OperationalError("no such table: memories")

    with pytest.raises(sqlite3.OperationalError):
        SqlWriteExecutor(FakeConns(conn), retry=BusyRetry(base_delay=0)).execute(work)
    assert seen["n"] == 1


def test_write_gives_up_after_max_attempts():
    conn = FakeConn()
    seen = {"n": 0}

    def work(c):
        seen["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError):
        SqlWriteExecutor(FakeConns(conn), retry=BusyRetry(max_attempts=3, base_delay=0)).execute(work)
    assert seen["n"] == 3


# --- read executor ---


def test_plain_read_runs_without_a_transaction():
    conn = FakeConn()
    assert SqlReadExecutor(FakeConns(conn)).execute(lambda c: "r") == "r"
    assert conn.calls == []


def test_snapshot_read_brackets_a_transaction():
    conn = FakeConn()
    executor = SqlReadExecutor(FakeConns(conn), strategy=SnapshotRead())
    assert executor.execute(lambda c: "r") == "r"
    assert conn.calls == ["BEGIN", "COMMIT"]


def test_snapshot_read_rolls_back_on_error():
    conn = FakeConn()

    def work(c):
        raise ValueError("x")

    with pytest.raises(ValueError):
        SqlReadExecutor(FakeConns(conn), strategy=SnapshotRead()).execute(work)
    assert conn.calls == ["BEGIN", "ROLLBACK"]


# --- retry policy logic ---


def test_busy_retry_policy_classification():
    policy = BusyRetry(max_attempts=3, base_delay=0)
    assert policy.should_retry(sqlite3.OperationalError("database is locked"), 1)
    assert policy.should_retry(sqlite3.OperationalError("database is busy"), 2)
    assert not policy.should_retry(sqlite3.OperationalError("locked"), 3)  # hit the cap
    assert not policy.should_retry(sqlite3.OperationalError("syntax error"), 1)


def test_plain_read_strategy_is_default():
    assert isinstance(SqlReadExecutor(FakeConns(FakeConn()))._strategy, PlainRead)
