"""SQLite + sqlite-vec + FTS5 store behind MemoryRepositoryPort.

One embedded file. The embedding is a ``BLOB`` column on the ``memories`` row,
ranked by the ``vec_distance_cosine`` scalar over a ``WHERE``-filtered scan (no
``vec0`` virtual table — so every structured filter, including the list-valued
``tags``/``related_files`` via ``json_each``, is a plain ``WHERE``, and the row
and its vector live together for single-table atomicity). FTS5 over ``content``
(external-content, trigger-synced) provides lexical search; hybrid retrieval is
reciprocal-rank fusion of the two ranked lists, computed in the adapter.

All SQL goes through two executors: the WRITE executor serialises through one
writer connection and runs each unit of work in a transaction (commit/rollback,
busy-retry); the READ executor runs reads on per-thread connections (WAL →
concurrent), and the two-leg hybrid retrieval runs under one snapshot. The
executors own every transaction; SqliteConnections is just the connection
resource. Rationale and alternatives weighed are in docs/adr/0001-storage-engine.md.
The schema is created lazily on the first write, once the vector dimension is known.
"""
from __future__ import annotations

import json
import re
import sqlite3

from sqlite_vec import serialize_float32

from mnemo.adapters.store.executors import SqlReadExecutor, SqlWriteExecutor
from mnemo.adapters.store.link_serializer import link_from_dict, link_to_dict
from mnemo.adapters.store.memory_serializer import from_dict, to_dict
from mnemo.adapters.store.rank_fusion import reciprocal_rank_fusion
from mnemo.adapters.store.sqlite_connections import SqliteConnections
from mnemo.adapters.store.transaction import SnapshotRead
from mnemo.application.retrieval import Retrieval
from mnemo.application.scored_memory import ScoredMemory
from mnemo.application.search_criteria import SearchCriteria
from mnemo.application.types import Vector
from mnemo.domain.link import Link
from mnemo.domain.memory import Memory

# Payload columns, qualified to the `m` alias so they stay unambiguous when the
# lexical leg joins the FTS table (which also has a `content` column).
_PAYLOAD = (
    "m.id, m.content, m.type, m.scope, m.project, m.tags, m.related_files,"
    " m.topic_key, m.session_id, m.status, m.supersedes, m.hash, m.created_at,"
    " m.updated_at"
)
# Over-fetch each retrieval leg before fusion so an item strong in one leg but
# outside the other's top-`limit` still survives. Brute-force makes this cheap.
_CANDIDATE_MULTIPLIER = 5

_INSERT_MEMORY = (
    "INSERT INTO memories (id, content, embedding, type, scope, project, tags,"
    " related_files, topic_key, session_id, status, supersedes, hash, created_at,"
    " updated_at) VALUES (:id, :content, :embedding,"
    " :type, :scope, :project, :tags, :related_files, :topic_key, :session_id,"
    " :status, :supersedes, :hash, :created_at, :updated_at)"
)


def _create_table_sql(name: str, dim: int) -> str:
    """The `memories` table at a given name and embedding dimension. The dimension is
    baked into the embedding-column CHECK; `set_dimension` builds a new table at the
    new dimension and swaps it in atomically."""
    return (
        f"CREATE TABLE {name} ("
        " id              TEXT PRIMARY KEY,"
        " content         TEXT NOT NULL,"
        f" embedding       BLOB CHECK(embedding IS NULL OR vec_length(embedding) == {dim}),"
        " type            TEXT NOT NULL,"
        " scope           TEXT NOT NULL,"
        " project         TEXT,"
        " tags            TEXT NOT NULL,"
        " related_files   TEXT NOT NULL,"
        " topic_key       TEXT,"
        " session_id      TEXT,"
        " status          TEXT NOT NULL,"
        " supersedes      TEXT,"
        " hash            TEXT NOT NULL,"
        " created_at      TEXT NOT NULL,"
        " updated_at      TEXT NOT NULL"
        ")"
    )


# Copy content + metadata into the rebuilt table, resetting every embedding to NULL
# (re-pending) so the caller re-embeds at the new dimension.
_COPY_TO_REBUILT = (
    "INSERT INTO memories_new (id, content, embedding, type, scope, project, tags,"
    " related_files, topic_key, session_id, status, supersedes, hash, created_at,"
    " updated_at) SELECT id, content, NULL, type, scope, project, tags, related_files,"
    " topic_key, session_id, status, supersedes, hash, created_at, updated_at FROM memories"
)

# Derived store state — built as part of schema CREATION so every store is correct
# from birth, and rebuilt verbatim after a `set_dimension` table swap (DB schema rule).
_INDEX_STATEMENTS = (
    "CREATE INDEX memories_hash ON memories(hash)",
    "CREATE INDEX memories_topic ON memories(topic_key, project)",
    "CREATE INDEX memories_status ON memories(status)",
)
_FTS_STATEMENT = (
    "CREATE VIRTUAL TABLE memories_fts USING fts5("
    " content, content='memories', content_rowid='rowid')"
)
_TRIGGER_STATEMENTS = (
    "CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN"
    " INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);"
    " END",
    "CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN"
    " INSERT INTO memories_fts(memories_fts, rowid, content)"
    " VALUES('delete', old.rowid, old.content);"
    " END",
    "CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN"
    " INSERT INTO memories_fts(memories_fts, rowid, content)"
    " VALUES('delete', old.rowid, old.content);"
    " INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);"
    " END",
)
# Repopulate the external-content FTS index from the (renamed) memories table.
_FTS_REBUILD = "INSERT INTO memories_fts(memories_fts) VALUES('rebuild')"


class SqliteVecMemoryRepository:
    def __init__(self, path: str, dim: int | None = None) -> None:
        # The embedding dimension. Given up front (from the embedder) it lets a
        # PENDING write — one with no vector yet, for deferred embedding — create the
        # schema before any vector exists. Left None, it is learned from the first
        # vectored add (the original behaviour), and a vector-less first write errors.
        self._dim = dim
        # The repository OWNS the connection resource and shares it between its two
        # executors; it talks to SQL only through them (the `_conns` handle is the
        # ownership reference, used for lifecycle).
        self._conns = SqliteConnections(path)
        self._read = SqlReadExecutor(self._conns)
        self._write = SqlWriteExecutor(self._conns)
        # The links table is dimension-independent, so it exists from the start
        # (an edge can be written before, or independently of, any memory row).
        self._write.execute(self._create_links_schema)
        self._ready = self._read.execute(self._memories_table_exists)

    @staticmethod
    def _create_links_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS links ("
            " source_id TEXT NOT NULL, target_id TEXT NOT NULL, type TEXT NOT NULL,"
            " provenance TEXT NOT NULL, created_at TEXT NOT NULL,"
            " PRIMARY KEY (source_id, target_id, type))"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS links_target ON links(target_id)")

    @staticmethod
    def _memories_table_exists(conn: sqlite3.Connection) -> bool:
        return (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'"
            ).fetchone()
            is not None
        )

    def add(self, memory: Memory, vector: Vector | None = None) -> None:
        # Fast path: the schema is known to exist, so just insert.
        if self._ready:
            self._write.execute(
                lambda conn: conn.execute(_INSERT_MEMORY, self._row(memory, vector))
            )
            return

        # First write(s): the existence check + create live INSIDE the unit of work,
        # under the writer lock, so concurrent first-writers can't both create the
        # table (the check is in-transaction, not the stale cached flag). The cached
        # `_ready` flag is flipped only AFTER the write commits.
        def work(conn: sqlite3.Connection) -> bool:
            if self._memories_table_exists(conn):
                conn.execute(_INSERT_MEMORY, self._row(memory, vector))
                return False
            dim = len(vector) if vector is not None else self._dim
            if dim is None:
                raise ValueError(
                    "cannot create the store schema without an embedding dimension:"
                    " pass dim= for a pending (vector-less) first write"
                )
            self._create_schema(conn, dim)
            conn.execute(_INSERT_MEMORY, self._row(memory, vector))
            return True

        created = self._write.execute(work)
        if created:
            self._ready = True

    def supersede(
        self, memory: Memory, link: Link, vector: Vector | None = None
    ) -> None:
        """Persist a supersede in ONE transaction — mark the prior (`memory.supersedes`)
        superseded, insert the successor, write the supersedes edge — so a crash can
        never leave the topic_key without an active record, or a successor without its
        edge. The caller owns the relationship (sets `memory.supersedes`, builds
        `link`); this only persists it atomically. The schema is assumed to exist (a
        supersede always replaces an existing row)."""

        def work(conn: sqlite3.Connection) -> None:
            self._mark_superseded(conn, memory.supersedes)
            conn.execute(_INSERT_MEMORY, self._row(memory, vector))
            self._insert_link(conn, link)

        self._write.execute(work)

    def set_dimension(self, new_dim: int) -> None:
        """Rebuild the store at a new embedding dimension (e.g. switching embedders).

        Content, metadata and links are preserved; every embedding is dropped to
        PENDING for re-computation by the caller. No-op if the dimension already
        matches.

        The rebuild is ATOMIC: it copies into a fresh table and swaps it in inside one
        transaction (the write executor's), so the original `memories` is never the only
        copy on disk and is dropped only once the rewrite has succeeded. A failure
        mid-rebuild rolls back to the original store intact; readers see the old table
        until commit, so the table never disappears under a concurrent query (DB schema
        rule: never drop/recreate to change a table).
        """
        if not self._ready:
            self._dim = new_dim  # fresh store — first write creates the schema at new_dim
            return
        if self._current_dim() == new_dim:
            return
        self._write.execute(lambda conn: self._rebuild_to(conn, new_dim))
        self._dim = new_dim

    def _rebuild_to(self, conn: sqlite3.Connection, new_dim: int) -> None:
        conn.execute(_create_table_sql("memories_new", new_dim))
        conn.execute(_COPY_TO_REBUILT)  # snapshot under the writer lock, in-txn
        conn.execute("DROP TRIGGER memories_ai")
        conn.execute("DROP TRIGGER memories_ad")
        conn.execute("DROP TRIGGER memories_au")
        conn.execute("DROP TABLE memories_fts")
        conn.execute("DROP TABLE memories")
        conn.execute("ALTER TABLE memories_new RENAME TO memories")
        self._rebuild_indexes_and_fts(conn)

    def _current_dim(self) -> int | None:
        """The dimension baked into the live schema (parsed from its CHECK clause)."""
        row = self._read.execute(
            lambda conn: conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories'"
            ).fetchone()
        )
        if not row or not row[0]:
            return None
        found = re.search(r"vec_length\(embedding\)\s*==\s*(\d+)", row[0])
        return int(found.group(1)) if found else None

    def set_vector(self, memory_id: str, vector: Vector) -> None:
        if not self._ready:
            return
        self._write.execute(
            lambda conn: conn.execute(
                "UPDATE memories SET embedding = ? WHERE id = ?",
                (serialize_float32(list(vector)), memory_id),
            )
        )

    def has_vector(self, memory_id: str) -> bool:
        if not self._ready:
            return False
        row = self._read.execute(
            lambda conn: conn.execute(
                "SELECT embedding IS NOT NULL FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        )
        return bool(row[0]) if row else False

    def content_for(self, memory_id: str) -> str | None:
        if not self._ready:
            return None
        row = self._read.execute(
            lambda conn: conn.execute(
                "SELECT content FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        )
        return row[0] if row else None

    def next_unembedded(self, limit: int) -> list[str]:
        if not self._ready or limit <= 0:
            return []
        rows = self._read.execute(
            lambda conn: conn.execute(
                "SELECT id FROM memories WHERE embedding IS NULL LIMIT ?", (limit,)
            ).fetchall()
        )
        return [row[0] for row in rows]

    def pending_count(self) -> int:
        if not self._ready:
            return 0
        (count,) = self._read.execute(
            lambda conn: conn.execute(
                "SELECT count(*) FROM memories WHERE embedding IS NULL"
            ).fetchone()
        )
        return count

    def find_active_by_hash(
        self, content_hash: str, project: str | None
    ) -> Memory | None:
        if project is None:
            predicate = "hash = ? AND status = 'active' AND project IS NULL"
            params: tuple = (content_hash,)
        else:
            predicate = "hash = ? AND status = 'active' AND project = ?"
            params = (content_hash, project)
        return self._read.execute(lambda conn: self._read_one(conn, predicate, params))

    def find_active_by_topic_key(
        self, topic_key: str, project: str | None
    ) -> Memory | None:
        if project is None:
            predicate = "topic_key = ? AND status = 'active' AND project IS NULL"
            params: tuple = (topic_key,)
        else:
            predicate = "topic_key = ? AND status = 'active' AND project = ?"
            params = (topic_key, project)
        return self._read.execute(lambda conn: self._read_one(conn, predicate, params))

    def retrieve(self, request: Retrieval) -> list[ScoredMemory]:
        limit = request.limit
        if not self._ready or limit <= 0:
            return []
        if request.text is None and request.vector is None:
            return self._browse(request.criteria, limit)
        where, params = self._where(request.criteria)
        candidate = limit * _CANDIDATE_MULTIPLIER
        match = _match_query(request.text)

        # Both legs run under ONE snapshot so a write committed between them cannot
        # make the dense and lexical lists disagree on which rows exist.
        def work(conn: sqlite3.Connection) -> tuple[list, list]:
            # Pending memories (embedding IS NULL, deferred embedding) have no vector
            # to rank — excluded from the dense leg; the lexical leg still finds them.
            dense = conn.execute(
                f"SELECT {_PAYLOAD}, vec_distance_cosine(m.embedding, ?) AS _distance"
                f" FROM memories m WHERE {where} AND m.embedding IS NOT NULL"
                f" ORDER BY _distance ASC LIMIT ?",
                (serialize_float32(list(request.vector)), *params, candidate),
            ).fetchall()
            lexical: list[sqlite3.Row] = []
            if match is not None:
                lexical = conn.execute(
                    f"SELECT {_PAYLOAD}, bm25(memories_fts) AS _rank FROM memories_fts"
                    f" JOIN memories m ON m.rowid = memories_fts.rowid"
                    f" WHERE memories_fts MATCH ? AND {where} ORDER BY _rank ASC LIMIT ?",
                    (match, *params, candidate),
                ).fetchall()
            return dense, lexical

        dense, lexical = self._read.execute(work, strategy=SnapshotRead())

        rows_by_id = {row["id"]: row for row in dense}
        for row in lexical:
            rows_by_id.setdefault(row["id"], row)

        fused = reciprocal_rank_fusion(
            [[row["id"] for row in dense], [row["id"] for row in lexical]]
        )
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [
            ScoredMemory(memory=self._to_memory(rows_by_id[memory_id]), score=score)
            for memory_id, score in ranked
        ]

    def _browse(self, criteria: SearchCriteria, limit: int) -> list[ScoredMemory]:
        """Filter-only retrieval, newest first — no query, no ranking. Pending
        (un-embedded) memories are included; the score is 0.0 (order conveys recency)."""
        where, params = self._where(criteria)
        rows = self._read.execute(
            lambda conn: conn.execute(
                f"SELECT {_PAYLOAD} FROM memories m WHERE {where}"
                f" ORDER BY m.created_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        )
        return [ScoredMemory(memory=self._to_memory(row), score=0.0) for row in rows]

    def mark_superseded(self, memory_id: str) -> None:
        self._write.execute(lambda conn: self._mark_superseded(conn, memory_id))

    def delete(self, ids: list[str]) -> int:
        if not self._ready or not ids:
            return 0
        placeholders = ", ".join("?" for _ in ids)

        def work(conn: sqlite3.Connection) -> int:
            conn.execute(
                f"DELETE FROM links WHERE source_id IN ({placeholders})"
                f" OR target_id IN ({placeholders})",
                (*ids, *ids),
            )
            return conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})", tuple(ids)
            ).rowcount

        return self._write.execute(work)

    def delete_by_project(self, project: str) -> int:
        if not self._ready:
            return 0

        def work(conn: sqlite3.Connection) -> int:
            conn.execute(
                "DELETE FROM links WHERE source_id IN"
                " (SELECT id FROM memories WHERE project = ?) OR target_id IN"
                " (SELECT id FROM memories WHERE project = ?)",
                (project, project),
            )
            return conn.execute(
                "DELETE FROM memories WHERE project = ?", (project,)
            ).rowcount

        return self._write.execute(work)

    def delete_all(self) -> int:
        def work(conn: sqlite3.Connection) -> int:
            conn.execute("DELETE FROM links")
            if not self._ready:
                return 0
            (removed,) = conn.execute("SELECT count(*) FROM memories").fetchone()
            conn.execute("DELETE FROM memories")
            return removed

        return self._write.execute(work)

    def list_all(self) -> list[Memory]:
        if not self._ready:
            return []
        rows = self._read.execute(
            lambda conn: conn.execute(f"SELECT {_PAYLOAD} FROM memories m").fetchall()
        )
        return [self._to_memory(row) for row in rows]

    def add_link(self, link: Link) -> None:
        self._write.execute(lambda conn: self._insert_link(conn, link))

    def links_for(self, memory_id: str) -> list[Link]:
        rows = self._read.execute(
            lambda conn: conn.execute(
                "SELECT source_id, target_id, type, provenance, created_at FROM links"
                " WHERE source_id = ? OR target_id = ?",
                (memory_id, memory_id),
            ).fetchall()
        )
        return [link_from_dict(dict(row)) for row in rows]

    # --- internals (pure SQL over a given connection; no transaction concern) ---

    def _mark_superseded(self, conn: sqlite3.Connection, memory_id: str) -> None:
        memory = self._read_one(conn, "id = ?", (memory_id,))
        if memory is None:
            return
        memory.mark_superseded()
        conn.execute(
            "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
            (memory.status, memory.updated_at, memory_id),
        )

    @staticmethod
    def _insert_link(conn: sqlite3.Connection, link: Link) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO links (source_id, target_id, type, provenance,"
            " created_at) VALUES (:source_id, :target_id, :type, :provenance, :created_at)",
            link_to_dict(link),
        )

    def _where(self, criteria: SearchCriteria) -> tuple[str, list[str]]:
        """Translate the criteria to a SQL WHERE, shared by both retrieval legs."""
        clauses = ["m.status = 'active'"]
        params: list[str] = []
        if criteria.scope == "global":
            clauses.append("m.scope = 'global'")
        elif criteria.scope != "all":  # 'project' = this project OR global (soft scope)
            # SearchCriteria guarantees a project when scope='project'.
            clauses.append("(m.project = ? OR m.scope = 'global')")
            params.append(criteria.project)
        if criteria.type is not None:
            clauses.append("m.type = ?")
            params.append(criteria.type.value)
        for tag in criteria.tags:  # ALL tags must be present
            clauses.append("EXISTS(SELECT 1 FROM json_each(m.tags) WHERE value = ?)")
            params.append(tag)
        if criteria.related_files:  # ANY of the files
            placeholders = ", ".join("?" for _ in criteria.related_files)
            clauses.append(
                "EXISTS(SELECT 1 FROM json_each(m.related_files)"
                f" WHERE value IN ({placeholders}))"
            )
            params.extend(criteria.related_files)
        if criteria.created_after is not None:
            clauses.append("m.created_at >= ?")
            params.append(criteria.created_after)
        return " AND ".join(clauses), params

    def _read_one(
        self, conn: sqlite3.Connection, predicate: str, params: tuple
    ) -> Memory | None:
        if not self._ready:
            return None
        row = conn.execute(
            f"SELECT {_PAYLOAD} FROM memories m WHERE {predicate} LIMIT 1", params
        ).fetchone()
        return self._to_memory(row) if row else None

    def _create_schema(self, conn: sqlite3.Connection, dim: int) -> None:
        """Pure SQL: create the memories table + derived state on the given (in-txn)
        connection. Does NOT commit and does NOT flip `_ready` — the caller sets the
        flag only after the write executor commits."""
        conn.execute(_create_table_sql("memories", dim))
        self._rebuild_indexes_and_fts(conn, rebuild=False)  # empty table — nothing to backfill

    def _rebuild_indexes_and_fts(
        self, conn: sqlite3.Connection, *, rebuild: bool = True
    ) -> None:
        """(Re)build the indexes, FTS5 table and sync triggers on `memories`. Run both
        when creating a fresh schema and as the final step of a `set_dimension` swap;
        `rebuild` backfills the external-content FTS index from the (renamed) table."""
        for statement in _INDEX_STATEMENTS:
            conn.execute(statement)
        conn.execute(_FTS_STATEMENT)
        for statement in _TRIGGER_STATEMENTS:
            conn.execute(statement)
        if rebuild:
            conn.execute(_FTS_REBUILD)

    def _row(self, memory: Memory, vector: Vector | None) -> dict:
        data = to_dict(memory)
        data["tags"] = json.dumps(data["tags"])
        data["related_files"] = json.dumps(data["related_files"])
        data["embedding"] = serialize_float32(list(vector)) if vector is not None else None
        return data

    @staticmethod
    def _to_memory(row: sqlite3.Row) -> Memory:
        return from_dict(
            {
                "id": row["id"],
                "content": row["content"],
                "type": row["type"],
                "scope": row["scope"],
                "project": row["project"],
                "related_files": json.loads(row["related_files"]),
                "tags": json.loads(row["tags"]),
                "topic_key": row["topic_key"],
                "session_id": row["session_id"],
                "status": row["status"],
                "supersedes": row["supersedes"],
                "hash": row["hash"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )


def _match_query(text: str | None) -> str | None:
    """Build a safe FTS5 MATCH string: each word token quoted, OR-joined.

    Quoting neutralizes FTS operators in user text; OR keeps lexical matching
    lenient (the dense leg handles semantic recall). Returns None when there is no
    query text or no usable tokens, so the lexical leg is simply skipped.
    """
    if not text:
        return None
    tokens = re.findall(r"\w+", text)
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens)
