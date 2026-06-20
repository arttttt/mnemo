"""SQL executors: the single owners of how work reaches the database.

A repository never touches a connection, the writer lock or ``BEGIN``/``COMMIT``
directly — it hands a *unit of work* (a callable over a connection that runs only
SQL) to an executor and gets the result back. The write executor serialises
through the one writer connection and runs the unit in a transaction (commit on
success, rollback on error, retry on a busy database); the read executor runs the
unit on a per-thread reader connection under a read strategy. This removes the old
split where the repository reached for ``SqliteConnections`` directly for some
calls and a transaction helper for others: now every SQL access — read or write —
goes through a symmetric ``executor.execute(work)``.

A unit of work MUST be pure SQL with no side effects outside the database: the
write executor may roll back and replay it (busy retry), and a read snapshot also
wraps it in a transaction. Caller-side effects (notify a scheduler, flip a cached
flag) belong AFTER ``execute`` returns, never inside the work.
"""
from __future__ import annotations

import sqlite3
from typing import Callable, Protocol, TypeVar

from mnemo.adapters.store.sqlite_connections import SqliteConnections
from mnemo.adapters.store.transaction import (
    BusyRetry,
    Immediate,
    PlainRead,
    ReadStrategy,
    RetryPolicy,
    WriteStrategy,
)

T = TypeVar("T")
Work = Callable[[sqlite3.Connection], T]


class Executor(Protocol):
    def execute(self, work: Work, *, strategy=None) -> T: ...


class SqlWriteExecutor:
    """Serialised, transactional writes. Holds the writer lock for the whole unit,
    opens a transaction per the write strategy, commits on success and rolls back
    on any error; retries the entire unit on a busy database per the retry policy."""

    def __init__(
        self,
        conns: SqliteConnections,
        *,
        strategy: WriteStrategy | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        self._conns = conns
        self._strategy = strategy or Immediate()
        self._retry = retry or BusyRetry()

    def execute(self, work: Work, *, strategy: WriteStrategy | None = None) -> T:
        begin = strategy or self._strategy
        attempt = 0
        while True:
            try:
                with self._conns.writer() as conn:
                    begin.begin(conn)
                    try:
                        result = work(conn)
                        conn.execute("COMMIT")
                        return result
                    except BaseException:
                        if conn.in_transaction:
                            conn.execute("ROLLBACK")
                        raise
            except sqlite3.OperationalError as error:
                # The lock is released (we left the `with`) and nothing is
                # committed (rolled back above), so a retry replays cleanly.
                attempt += 1
                if self._retry.should_retry(error, attempt):
                    self._retry.backoff(attempt)
                    continue
                raise


class SqlReadExecutor:
    """Concurrent reads over per-thread reader connections (WAL). Runs the unit
    under a read strategy — plain by default, or a consistent snapshot for
    multi-statement reads that must agree on one view of the store."""

    def __init__(
        self, conns: SqliteConnections, *, strategy: ReadStrategy | None = None
    ) -> None:
        self._conns = conns
        self._strategy = strategy or PlainRead()

    def execute(self, work: Work, *, strategy: ReadStrategy | None = None) -> T:
        return (strategy or self._strategy).run(self._conns.reader(), work)
