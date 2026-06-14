"""SQLite + sqlite-vec + FTS5 store behind MemoryRepositoryPort.

One embedded file. The embedding is a ``BLOB`` column on the ``memories`` row,
ranked by the ``vec_distance_cosine`` scalar over a ``WHERE``-filtered scan (no
``vec0`` virtual table — so every structured filter, including the list-valued
``tags``/``related_files`` via ``json_each``, is a plain ``WHERE``, and the row
and its vector live together for single-table atomicity). FTS5 over ``content``
(external-content, trigger-synced) provides lexical search; hybrid retrieval is
reciprocal-rank fusion of the two ranked lists, computed in the adapter.

Rationale and the alternatives weighed are in docs/adr/0001-storage-engine.md.
The schema is created lazily on the first write, once the vector dimension is
known (mirroring the LanceDB adapter it replaces).
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import sqlite_vec
from sqlite_vec import serialize_float32

from mnemo.adapters.store.memory_serializer import from_dict, to_dict
from mnemo.adapters.store.rank_fusion import reciprocal_rank_fusion
from mnemo.application.scored_memory import ScoredMemory
from mnemo.application.search_criteria import SearchCriteria
from mnemo.application.types import Vector
from mnemo.domain.memory import Memory

# Payload columns, qualified to the `m` alias so they stay unambiguous when the
# lexical leg joins the FTS table (which also has a `content` column).
_PAYLOAD = (
    "m.id, m.content, m.type, m.scope, m.project, m.tags, m.related_files,"
    " m.topic_key, m.session_id, m.status, m.supersedes, m.hash, m.created_at,"
    " m.updated_at, m.last_seen_at, m.duplicate_count"
)
# Over-fetch each retrieval leg before fusion so an item strong in one leg but
# outside the other's top-`limit` still survives. Brute-force makes this cheap.
_CANDIDATE_MULTIPLIER = 5


class SqliteVecMemoryRepository:
    def __init__(self, path: str) -> None:
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        for pragma in (
            "journal_mode=WAL",
            "synchronous=NORMAL",
            "busy_timeout=5000",
            "foreign_keys=ON",
        ):
            self._conn.execute(f"PRAGMA {pragma}")
        self._ready = (
            self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'"
            ).fetchone()
            is not None
        )

    def add(self, memory: Memory, vector: Vector) -> None:
        if not self._ready:
            self._create_schema(len(vector))
        self._conn.execute(
            "INSERT INTO memories (id, content, embedding, type, scope, project, tags,"
            " related_files, topic_key, session_id, status, supersedes, hash, created_at,"
            " updated_at, last_seen_at, duplicate_count) VALUES (:id, :content, :embedding,"
            " :type, :scope, :project, :tags, :related_files, :topic_key, :session_id,"
            " :status, :supersedes, :hash, :created_at, :updated_at, :last_seen_at,"
            " :duplicate_count)",
            self._row(memory, vector),
        )
        self._conn.commit()

    def find_by_hash(self, content_hash: str) -> Memory | None:
        return self._first("hash = ?", (content_hash,))

    def find_active_by_topic_key(
        self, topic_key: str, project: str | None
    ) -> Memory | None:
        if project is None:
            return self._first(
                "topic_key = ? AND status = 'active' AND project IS NULL", (topic_key,)
            )
        return self._first(
            "topic_key = ? AND status = 'active' AND project = ?", (topic_key, project)
        )

    def search(
        self, query: str, vector: Vector, criteria: SearchCriteria, limit: int
    ) -> list[ScoredMemory]:
        if not self._ready or limit <= 0:
            return []
        where, params = self._where(criteria)
        candidate = limit * _CANDIDATE_MULTIPLIER

        dense = self._conn.execute(
            f"SELECT {_PAYLOAD}, vec_distance_cosine(m.embedding, ?) AS _distance"
            f" FROM memories m WHERE {where} ORDER BY _distance ASC LIMIT ?",
            (serialize_float32(list(vector)), *params, candidate),
        ).fetchall()

        lexical: list[sqlite3.Row] = []
        match = _match_query(query)
        if match is not None:
            lexical = self._conn.execute(
                f"SELECT {_PAYLOAD}, bm25(memories_fts) AS _rank FROM memories_fts"
                f" JOIN memories m ON m.rowid = memories_fts.rowid"
                f" WHERE memories_fts MATCH ? AND {where} ORDER BY _rank ASC LIMIT ?",
                (match, *params, candidate),
            ).fetchall()

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

    def register_duplicate(self, memory_id: str) -> None:
        memory = self._by_id(memory_id)
        if memory is None:
            return
        memory.register_duplicate()
        self._conn.execute(
            "UPDATE memories SET duplicate_count = ?, last_seen_at = ? WHERE id = ?",
            (memory.duplicate_count, memory.last_seen_at, memory_id),
        )
        self._conn.commit()

    def mark_superseded(self, memory_id: str) -> None:
        memory = self._by_id(memory_id)
        if memory is None:
            return
        memory.mark_superseded()
        self._conn.execute(
            "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
            (memory.status, memory.updated_at, memory_id),
        )
        self._conn.commit()

    def delete(self, ids: list[str]) -> int:
        if not self._ready or not ids:
            return 0
        placeholders = ", ".join("?" for _ in ids)
        cursor = self._conn.execute(
            f"DELETE FROM memories WHERE id IN ({placeholders})", tuple(ids)
        )
        self._conn.commit()
        return cursor.rowcount

    def delete_by_project(self, project: str) -> int:
        if not self._ready:
            return 0
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE project = ?", (project,)
        )
        self._conn.commit()
        return cursor.rowcount

    def delete_all(self) -> int:
        if not self._ready:
            return 0
        (removed,) = self._conn.execute("SELECT count(*) FROM memories").fetchone()
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()
        return removed

    def list_all(self) -> list[Memory]:
        if not self._ready:
            return []
        rows = self._conn.execute(f"SELECT {_PAYLOAD} FROM memories m").fetchall()
        return [self._to_memory(row) for row in rows]

    # --- internals ---

    def _where(self, criteria: SearchCriteria) -> tuple[str, list[str]]:
        """Translate the criteria to a SQL WHERE, shared by both retrieval legs."""
        clauses = ["m.status = 'active'"]
        params: list[str] = []
        if criteria.scope == "global":
            clauses.append("m.scope = 'global'")
        elif criteria.scope != "all":  # 'project' = this project OR global (soft scope)
            if criteria.project is None:
                clauses.append("(m.project IS NULL OR m.scope = 'global')")
            else:
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

    def _first(self, predicate: str, params: tuple) -> Memory | None:
        if not self._ready:
            return None
        row = self._conn.execute(
            f"SELECT {_PAYLOAD} FROM memories m WHERE {predicate} LIMIT 1", params
        ).fetchone()
        return self._to_memory(row) if row else None

    def _by_id(self, memory_id: str) -> Memory | None:
        return self._first("id = ?", (memory_id,))

    def _create_schema(self, dim: int) -> None:
        self._conn.executescript(
            f"""
            CREATE TABLE memories (
                id              TEXT PRIMARY KEY,
                content         TEXT NOT NULL,
                embedding       BLOB NOT NULL CHECK(vec_length(embedding) == {dim}),
                type            TEXT NOT NULL,
                scope           TEXT NOT NULL,
                project         TEXT,
                tags            TEXT NOT NULL,
                related_files   TEXT NOT NULL,
                topic_key       TEXT,
                session_id      TEXT,
                status          TEXT NOT NULL,
                supersedes      TEXT,
                hash            TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                last_seen_at    TEXT NOT NULL,
                duplicate_count INTEGER NOT NULL
            );
            CREATE INDEX memories_hash ON memories(hash);
            CREATE INDEX memories_topic ON memories(topic_key, project);
            CREATE INDEX memories_status ON memories(status);

            CREATE VIRTUAL TABLE memories_fts USING fts5(
                content, content='memories', content_rowid='rowid'
            );
            CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content)
                VALUES('delete', old.rowid, old.content);
            END;
            CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content)
                VALUES('delete', old.rowid, old.content);
                INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            """
        )
        self._conn.commit()
        self._ready = True

    def _row(self, memory: Memory, vector: Vector) -> dict:
        data = to_dict(memory)
        data["tags"] = json.dumps(data["tags"])
        data["related_files"] = json.dumps(data["related_files"])
        data["embedding"] = serialize_float32(list(vector))
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
                "last_seen_at": row["last_seen_at"],
                "duplicate_count": row["duplicate_count"],
            }
        )


def _match_query(text: str) -> str | None:
    """Build a safe FTS5 MATCH string: each word token quoted, OR-joined.

    Quoting neutralizes FTS operators in user text; OR keeps lexical matching
    lenient (the dense leg handles semantic recall). Returns None when the query
    has no usable tokens, so the lexical leg is simply skipped.
    """
    tokens = re.findall(r"\w+", text)
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens)
