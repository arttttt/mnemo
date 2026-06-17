"""The one-off store migration (drop_dedup_columns).

DISPOSABLE: delete together with the migration once every live store is migrated.
Exercises the real DROP COLUMN against the `vec_length` CHECK constraint, which is
why the migration must load sqlite-vec.
"""
import sqlite3

import pytest

pytest.importorskip("sqlite_vec")
import sqlite_vec

from mnemo.infrastructure.migrations import drop_dedup_columns


def _legacy_store(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY, content TEXT NOT NULL,
            embedding BLOB CHECK(embedding IS NULL OR vec_length(embedding) == 4),
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL, duplicate_count INTEGER NOT NULL
        );
        INSERT INTO memories VALUES ('m1', 'hello', NULL, 't', 't', 't', 3);
        """
    )
    conn.commit()
    conn.close()


def test_drops_legacy_columns_and_keeps_data(tmp_path):
    db = str(tmp_path / "memory.db")
    _legacy_store(db)

    assert set(drop_dedup_columns(db)) == {"duplicate_count", "last_seen_at"}

    conn = sqlite3.connect(db)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
    assert "duplicate_count" not in cols and "last_seen_at" not in cols
    assert conn.execute("SELECT content FROM memories WHERE id='m1'").fetchone()[0] == "hello"
    conn.close()


def test_is_idempotent(tmp_path):
    db = str(tmp_path / "memory.db")
    _legacy_store(db)
    drop_dedup_columns(db)
    assert drop_dedup_columns(db) == []  # already migrated → no-op


def test_no_op_on_absent_or_empty_store(tmp_path):
    assert drop_dedup_columns(str(tmp_path / "missing.db")) == []
    empty = str(tmp_path / "empty.db")
    sqlite3.connect(empty).close()  # a db file with no `memories` table
    assert drop_dedup_columns(empty) == []
