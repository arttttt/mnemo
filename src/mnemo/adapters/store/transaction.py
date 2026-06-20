"""Transaction strategies for the SQL executors.

An executor owns the transaction lifecycle; a *strategy* decides HOW a unit of
work is bracketed. Write side: which ``BEGIN`` flavour to open with, and whether
to retry a busy database. Read side: whether the work runs under one consistent
snapshot. Keeping these as small, swappable objects lets the policy change
without touching either the executor or the repository's operations.
"""
from __future__ import annotations

import sqlite3
import time
from typing import Callable, Protocol, TypeVar

T = TypeVar("T")


# --- write strategies: how to begin the unit ---


class WriteStrategy(Protocol):
    def begin(self, conn: sqlite3.Connection) -> None: ...


class Immediate:
    """``BEGIN IMMEDIATE`` — take the write lock at the start of the unit, so two
    writers can never both read-then-upgrade into a deadlock. The default for writes."""

    def begin(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")


class Deferred:
    """``BEGIN`` — the write lock is acquired lazily on the first write statement.
    Available for read-mostly units where the upgrade race cannot occur."""

    def begin(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN")


# --- retry policy: what to do when the database is busy ---


class RetryPolicy(Protocol):
    def should_retry(self, error: sqlite3.OperationalError, attempt: int) -> bool: ...

    def backoff(self, attempt: int) -> None: ...


class BusyRetry:
    """Retry the WHOLE unit of work on a busy/locked database, with exponential
    backoff, up to ``max_attempts`` tries. Only ``SQLITE_BUSY``/``LOCKED`` is
    retried — any other ``OperationalError`` (a genuine error: no such table,
    constraint violation, ...) propagates at once. Safe because a work-unit is
    pure SQL inside one transaction: a rolled-back attempt leaves nothing behind
    to replay incorrectly.
    """

    def __init__(self, max_attempts: int = 5, base_delay: float = 0.05) -> None:
        self._max_attempts = max_attempts
        self._base_delay = base_delay

    def should_retry(self, error: sqlite3.OperationalError, attempt: int) -> bool:
        message = str(error).lower()
        transient = "locked" in message or "busy" in message
        return transient and attempt < self._max_attempts

    def backoff(self, attempt: int) -> None:
        time.sleep(self._base_delay * (2 ** (attempt - 1)))


class NoRetry:
    """Never retry — failures surface on the first attempt."""

    def should_retry(self, error: sqlite3.OperationalError, attempt: int) -> bool:
        return False

    def backoff(self, attempt: int) -> None:  # pragma: no cover - never reached
        return None


# --- read strategies: under what isolation the read runs ---


class ReadStrategy(Protocol):
    def run(self, conn: sqlite3.Connection, work: Callable[[sqlite3.Connection], T]) -> T: ...


class PlainRead:
    """Run the read with no explicit transaction. The default for single-statement
    reads — SQLite already gives each statement a consistent view."""

    def run(self, conn: sqlite3.Connection, work: Callable[[sqlite3.Connection], T]) -> T:
        return work(conn)


class SnapshotRead:
    """Bracket the read in a transaction so several statements observe ONE
    consistent snapshot — e.g. the dense and lexical legs of a hybrid retrieval,
    which must agree on what rows exist even if a write commits between them."""

    def run(self, conn: sqlite3.Connection, work: Callable[[sqlite3.Connection], T]) -> T:
        conn.execute("BEGIN")
        try:
            result = work(conn)
            conn.execute("COMMIT")
            return result
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
