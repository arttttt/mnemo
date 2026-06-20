"""SQLite + sqlite-vec + FTS5 store — realizes the memory store ports
(MemoryRepository + EmbeddingQueue) over one DB.

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

The repository is STATELESS: the embedding dimension comes from config (passed in)
and the schema is ensured EAGERLY at construction (= service/CLI startup). The
authoritative state lives in SQLite — the store is always ready after __init__, so
there is no cached `ready`/`dim` flag and no lazy first-write schema path.
"""
from __future__ import annotations

import json
import re
import sqlite3

from sqlite_vec import serialize_float32

from mnemo.adapters.store.executors import SqlReadExecutor, SqlWriteExecutor
from mnemo.adapters.store.memory_serializer import from_dict, to_dict
from mnemo.adapters.store.rank_fusion import reciprocal_rank_fusion
from mnemo.adapters.store.sqlite_connections import SqliteConnections
from mnemo.adapters.store.transaction import SnapshotRead
from mnemo.application.retrieval import Retrieval
from mnemo.application.scored_memory import ScoredMemory
from mnemo.application.search_criteria import SearchCriteria
from mnemo.application.types import Vector
from mnemo.domain.generators import now
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
    new dimension and swaps it in atomically.

    `project` foreign-keys to `projects(slug)` ON DELETE CASCADE — so deleting a
    project atomically deletes its memories (and, via the links FK, their edges),
    and a memory can only be written for a registered project (NULL = global,
    which the FK allows). Requires the `projects` table to exist first (the
    composition root builds the registry before the store)."""
    return (
        f"CREATE TABLE {name} ("
        " id              TEXT PRIMARY KEY,"
        " content         TEXT NOT NULL,"
        f" embedding       BLOB CHECK(embedding IS NULL OR vec_length(embedding) == {dim}),"
        " type            TEXT NOT NULL,"
        " scope           TEXT NOT NULL,"
        " project         TEXT REFERENCES projects(slug) ON DELETE CASCADE,"
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


class SqliteRepositoryImpl:
    def __init__(self, connections: SqliteConnections, dim: int) -> None:
        # The connection resource is INJECTED so the project registry can share the same
        # DB (one writer, one lock) — required for the FK cascade to run atomically.
        # The embedding dimension comes from config (the embedder) — the repository does
        # not learn or cache it. It is used once here to create the schema; the live
        # dimension thereafter is read from the schema itself (see _current_dim).
        self._read = SqlReadExecutor(connections)
        self._write = SqlWriteExecutor(connections)
        # Ensure the schema EAGERLY at construction (= service/CLI startup): the links
        # table (dimension-independent) plus the memories table at `dim` if it is
        # absent. After this the store is always ready, so no method needs an
        # existence flag and there is no lazy first-write schema path. An existing
        # store at a different dimension is left untouched here — `set_dimension`
        # (reindex) migrates it explicitly.
        self._write.execute(lambda conn: self._ensure_schema(conn, dim))

    @classmethod
    def open(cls, path: str, dim: int) -> "SqliteRepositoryImpl":
        """Build a store that owns its connection — for standalone CLI/tests. The
        composition root instead injects a SHARED SqliteConnections so the project
        registry lives in the same DB."""
        return cls(SqliteConnections(path), dim)

    def _ensure_schema(self, conn: sqlite3.Connection, dim: int) -> None:
        if not self._memories_table_exists(conn):
            self._create_schema(conn, dim)

    @staticmethod
    def _memories_table_exists(conn: sqlite3.Connection) -> bool:
        return (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'"
            ).fetchone()
            is not None
        )

    def add(self, memory: Memory, vector: Vector | None = None) -> None:
        # The schema exists (ensured at construction), so a write is just an insert.
        # `vector=None` stores the row pending (lexically searchable, absent from the
        # dense leg) until set_vector lands.
        self._write.execute(lambda conn: self._insert_memory(conn, memory, vector))

    def supersede(self, memory: Memory, vector: Vector | None = None) -> None:
        """Persist a supersede in ONE transaction — mark the prior (`memory.supersedes`)
        superseded and insert the successor — so a crash can never leave the topic_key
        without an active record. The caller owns the relationship (sets
        `memory.supersedes`); this only persists it atomically. The supersede link lives
        in the `supersedes` column itself."""

        def work(conn: sqlite3.Connection) -> None:
            self._mark_superseded(conn, memory.supersedes)
            self._insert_memory(conn, memory, vector)

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
        if self._current_dim() == new_dim:
            return
        self._write.execute(lambda conn: self._rebuild_to(conn, new_dim))

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
        self._write.execute(
            lambda conn: conn.execute(
                "UPDATE memories SET embedding = ? WHERE id = ?",
                (serialize_float32(list(vector)), memory_id),
            )
        )

    def has_vector(self, memory_id: str) -> bool:
        row = self._read.execute(
            lambda conn: conn.execute(
                "SELECT embedding IS NOT NULL FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        )
        return bool(row[0]) if row else False

    def content_for(self, memory_id: str) -> str | None:
        row = self._read.execute(
            lambda conn: conn.execute(
                "SELECT content FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        )
        return row[0] if row else None

    def next_unembedded(self, limit: int) -> list[str]:
        if limit <= 0:
            return []
        rows = self._read.execute(
            lambda conn: conn.execute(
                "SELECT id FROM memories WHERE embedding IS NULL LIMIT ?", (limit,)
            ).fetchall()
        )
        return [row[0] for row in rows]

    def pending_count(self) -> int:
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
        if limit <= 0:
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

    def delete(self, ids: list[str]) -> int:
        if not ids:
            return 0
        placeholders = ", ".join("?" for _ in ids)
        return self._write.execute(
            lambda conn: conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})", tuple(ids)
            ).rowcount
        )

    def delete_all(self) -> int:
        def work(conn: sqlite3.Connection) -> int:
            (removed,) = conn.execute("SELECT count(*) FROM memories").fetchone()
            conn.execute("DELETE FROM memories")
            return removed

        return self._write.execute(work)

    def list_all(self) -> list[Memory]:
        rows = self._read.execute(
            lambda conn: conn.execute(f"SELECT {_PAYLOAD} FROM memories m").fetchall()
        )
        return [self._to_memory(row) for row in rows]

    # --- internals (pure SQL over a given connection; no transaction concern) ---

    def _insert_memory(
        self, conn: sqlite3.Connection, memory: Memory, vector: Vector | None
    ) -> None:
        conn.execute(_INSERT_MEMORY, self._row(memory, vector))

    @staticmethod
    def _mark_superseded(conn: sqlite3.Connection, memory_id: str) -> None:
        # Persist the supersede transition with a direct UPDATE: no Memory is
        # reconstructed and none is mutated. The status vocabulary already lives in this
        # adapter (the `status = 'active'` retrieval filter); `updated_at` is bumped as
        # the domain transition does. A no-op if the id is gone.
        conn.execute(
            "UPDATE memories SET status = 'superseded', updated_at = ? WHERE id = ?",
            (now(), memory_id),
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
        row = conn.execute(
            f"SELECT {_PAYLOAD} FROM memories m WHERE {predicate} LIMIT 1", params
        ).fetchone()
        return self._to_memory(row) if row else None

    def _create_schema(self, conn: sqlite3.Connection, dim: int) -> None:
        """Pure SQL: create the memories table + derived state on the given (in-txn)
        connection. Does NOT commit — the write executor owns the transaction."""
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
