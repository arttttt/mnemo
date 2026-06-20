"""One-shot, DISPOSABLE startup migration: add the project/link foreign keys.

The fresh-store schema already declares the FKs (see sqlite_vec_repository), so this
only upgrades a user's pre-existing **pre-FK** store. It runs automatically before the
store is opened (in `build_container`, so BOTH the service and the CLI apply it — the
CLI's gate needs the seeded registry too, not just the service). It is idempotent.

It (1) seeds `projects` from the distinct `project` values already in `memories`
(plus the reserved global sentinel), then (2) rebuilds `memories` and `links` WITH the
foreign keys via an atomic copy→swap inside one transaction — the original tables are
never the only copy on disk, so any failure rolls back to the intact store.

DELETE this module and its call in `build_container` once the live store is migrated
(per the DB migration policy: a disposable migration, not a permanent runtime check).
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3

import sqlite_vec

from mnemo.adapters.store.sqlite_vec_repository import (
    _FTS_STATEMENT,
    _INDEX_STATEMENTS,
    _TRIGGER_STATEMENTS,
    _create_table_sql,
)
from mnemo.domain.constants import GLOBAL_PROJECT
from mnemo.domain.generators import now

_log = logging.getLogger("mnemo.migrations")

# memories columns in schema order — copied verbatim (embeddings preserved, unlike a
# reindex which resets them, because the dimension is unchanged here).
_MEMORY_COLUMNS = (
    "id, content, embedding, type, scope, project, tags, related_files, topic_key,"
    " session_id, status, supersedes, hash, created_at, updated_at"
)


def _links_ddl(name: str) -> str:
    """links WITH the FK — mirrors SqliteRepositoryImpl._create_links_schema, named so
    the rebuilt table can be created alongside the old one before the swap."""
    return (
        f"CREATE TABLE {name} ("
        " source_id TEXT NOT NULL, target_id TEXT NOT NULL, type TEXT NOT NULL,"
        " provenance TEXT NOT NULL, created_at TEXT NOT NULL,"
        " PRIMARY KEY (source_id, target_id, type),"
        " FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,"
        " FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE)"
    )


def add_project_foreign_keys(sqlite_path: str) -> bool:
    """Upgrade a pre-FK store in place. Returns True if it migrated, False on a no-op
    (no store yet, or already migrated)."""
    if not sqlite_path or not os.path.exists(sqlite_path):
        return False  # fresh install — the eager schema already declares the FKs
    conn = sqlite3.connect(sqlite_path)
    conn.isolation_level = None  # we drive the transaction explicitly
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)  # the memories CHECK(vec_length(...)) needs the extension
    conn.enable_load_extension(False)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        if not _has_table(conn, "memories") or _already_migrated(conn):
            return False
        _rebuild_with_fks(conn)
        _log.info("migrated store %s: added project/link foreign keys", sqlite_path)
        return True
    finally:
        conn.close()


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def _already_migrated(conn: sqlite3.Connection) -> bool:
    (sql,) = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories'"
    ).fetchone()
    return "REFERENCES projects" in (sql or "")


def _dim(conn: sqlite3.Connection) -> int:
    (sql,) = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories'"
    ).fetchone()
    found = re.search(r"vec_length\(embedding\)\s*==\s*(\d+)", sql or "")
    if not found:
        raise RuntimeError("cannot determine the store's embedding dimension to migrate")
    return int(found.group(1))


def _rebuild_with_fks(conn: sqlite3.Connection) -> None:
    dim = _dim(conn)
    timestamp = now()
    # FKs OFF during the rebuild so DROP TABLE doesn't cascade mid-swap; the seeded
    # registry makes the existing rows consistent, and the live service reopens with
    # foreign_keys=ON to enforce them going forward. (Cannot toggle inside a txn.)
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("BEGIN IMMEDIATE")
    try:
        # 1. The registry must exist (and hold every used slug) before the memories FK.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS projects"
            " (slug TEXT PRIMARY KEY, description TEXT, created_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO projects (slug, description, created_at) VALUES (?, NULL, ?)",
            (GLOBAL_PROJECT, timestamp),
        )
        conn.execute(
            "INSERT OR IGNORE INTO projects (slug, description, created_at)"
            " SELECT DISTINCT project, NULL, ? FROM memories WHERE project IS NOT NULL",
            (timestamp,),
        )
        # 2. memories WITH the FK; embeddings preserved (same dimension).
        conn.execute(_create_table_sql("memories_new", dim))
        conn.execute(
            f"INSERT INTO memories_new ({_MEMORY_COLUMNS})"
            f" SELECT {_MEMORY_COLUMNS} FROM memories"
        )
        # 3. links WITH the FK (a very old store may predate the links table).
        conn.execute(_links_ddl("links_new"))
        if _has_table(conn, "links"):
            conn.execute(
                "INSERT INTO links_new (source_id, target_id, type, provenance, created_at)"
                " SELECT source_id, target_id, type, provenance, created_at FROM links"
            )
        # 4. Drop old derived state + tables; swap the rebuilt ones in.
        for trigger in ("memories_ai", "memories_ad", "memories_au"):
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        conn.execute("DROP TABLE IF EXISTS memories_fts")
        conn.execute("DROP TABLE memories")
        conn.execute("DROP TABLE IF EXISTS links")
        conn.execute("ALTER TABLE memories_new RENAME TO memories")
        conn.execute("ALTER TABLE links_new RENAME TO links")
        # 5. Rebuild indexes, FTS and triggers — identical to the fresh schema.
        for statement in _INDEX_STATEMENTS:
            conn.execute(statement)
        conn.execute(_FTS_STATEMENT)
        for statement in _TRIGGER_STATEMENTS:
            conn.execute(statement)
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        conn.execute("CREATE INDEX IF NOT EXISTS links_target ON links(target_id)")
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
