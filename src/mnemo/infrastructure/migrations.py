"""One-off, disposable store migrations — run once at service start.

DISPOSABLE per the DB migration policy: each migration is idempotent and exists
only to bring an existing store up to the current schema. New stores are created
correct from birth, so once every live store has been migrated, delete the
migration and its call site (in the service) in a follow-up.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

# The exact-dedup accounting (duplicate_count / last_seen_at) was write-only —
# tracked but never read — so the columns are dropped. The current schema never
# creates them; this brings a pre-existing store in line.
_LEGACY_DEDUP_COLUMNS = ("duplicate_count", "last_seen_at")


def drop_dedup_columns(db_path: str) -> list[str]:
    """Drop the unused exact-dedup accounting columns from an existing store.

    Idempotent and safe when the store file is absent, has no `memories` table, or
    the columns are already gone. Returns the columns actually dropped.
    """
    if not Path(db_path).expanduser().exists():
        return []
    conn = _connect(db_path)
    try:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'"
        ).fetchone()
        if has_table is None:
            return []
        present = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
        dropped: list[str] = []
        for column in _LEGACY_DEDUP_COLUMNS:
            if column in present:
                conn.execute(f"ALTER TABLE memories DROP COLUMN {column}")
                dropped.append(column)
        conn.commit()
        return dropped
    finally:
        conn.close()


def _connect(db_path: str) -> sqlite3.Connection:
    # The `memories` schema carries a CHECK(vec_length(embedding) ...) constraint,
    # so the sqlite-vec function must be registered for the table rewrite a
    # DROP COLUMN performs.
    conn = sqlite3.connect(str(Path(db_path).expanduser()))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn
