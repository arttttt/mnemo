"""SQLite connection management: one serialized writer + per-thread readers.

SQLite in WAL mode allows a single writer and many concurrent readers. So all
writes share one writer connection guarded by a lock (serializing them before
they reach SQLite, avoiding SQLITE_BUSY), while each thread gets its own reader
connection — readers never block one another or the writer. This keeps the
repository's query methods free of any connection-management concern.
"""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec


class SqliteConnections:
    def __init__(self, path: str) -> None:
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._write_lock = threading.Lock()
        self._writer = self._open()
        self._reader_local = threading.local()
        # Per-thread readers live in thread-local storage, which can't be enumerated;
        # keep a flat registry too so close() can release every one (and its -wal/-shm).
        self._readers: list[sqlite3.Connection] = []
        self._readers_lock = threading.Lock()

    @contextmanager
    def writer(self) -> Iterator[sqlite3.Connection]:
        """The single writer connection, held under a lock for the call."""
        with self._write_lock:
            yield self._writer

    def reader(self) -> sqlite3.Connection:
        """A read connection private to the calling thread (WAL → concurrent)."""
        conn = getattr(self._reader_local, "conn", None)
        if conn is None:
            conn = self._open()
            self._reader_local.conn = conn
            with self._readers_lock:
                self._readers.append(conn)
        return conn

    def close(self) -> None:
        """Close the writer and every reader connection. Closing the writer lets SQLite
        checkpoint and drop the -wal/-shm sidecars. Call on a clean shutdown; after it
        the instance must not be used again."""
        with self._write_lock:
            self._writer.close()
        with self._readers_lock:
            for conn in self._readers:
                conn.close()
            self._readers.clear()

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        for pragma in (
            "journal_mode=WAL",
            "synchronous=NORMAL",
            "busy_timeout=5000",
            "foreign_keys=ON",
        ):
            conn.execute(f"PRAGMA {pragma}")
        return conn
