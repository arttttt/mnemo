"""LanceDB-backed repository: real ANN + persistence behind MemoryRepositoryPort.

The scaling store for the memory layer. It implements MemoryRepositoryPort
structurally; the in-memory store stays the offline/test default. The table is
created lazily on the first write, once the embedding dimension is known.
"""
from __future__ import annotations

from mnemo.adapters.store.memory_serializer import from_dict, to_dict
from mnemo.application.scored_memory import ScoredMemory
from mnemo.application.search_criteria import SearchCriteria
from mnemo.application.types import Vector
from mnemo.domain.memory import Memory

_TABLE = "memories"


class LanceDbMemoryRepository:
    def __init__(self, uri: str) -> None:
        import lancedb  # heavy, optional dependency — import lazily

        self._db = lancedb.connect(uri)
        self._fts_ready = False
        try:
            self._table = self._db.open_table(_TABLE)
        except ValueError:
            self._table = None  # created on the first add, when the dim is known

    def add(self, memory: Memory, vector: Vector) -> None:
        table = self._ensure_table(len(vector))
        table.add([{**to_dict(memory), "vector": list(vector)}])

    def find_by_hash(self, content_hash: str) -> Memory | None:
        return self._first(f"hash = {self._quote(content_hash)}")

    def find_active_by_topic_key(
        self, topic_key: str, project: str | None
    ) -> Memory | None:
        if self._table is None:
            return None
        predicate = f"topic_key = {self._quote(topic_key)} AND status = 'active'"
        matches = self._table.count_rows(filter=predicate)
        if matches == 0:
            return None
        for row in self._table.search().where(predicate).limit(matches).to_list():
            if row["project"] == project:
                return from_dict(row)
        return None

    def search(
        self, query: str, vector: Vector, criteria: SearchCriteria, limit: int
    ) -> list[ScoredMemory]:
        if self._table is None:
            return []
        self._ensure_fts_index()
        rows = (
            self._table.search(query_type="hybrid")
            .vector(list(vector))
            .text(query)
            .where(self._where(criteria), prefilter=True)
            .limit(limit)
            .to_list()
        )
        # Native hybrid fuses dense + full-text via reciprocal-rank fusion.
        return [
            ScoredMemory(memory=from_dict(row), score=row["_relevance_score"])
            for row in rows
        ]

    def _ensure_fts_index(self) -> None:
        # The full-text index backs the lexical half of hybrid search. Create it
        # once if absent — this also upgrades a table written before FTS existed
        # (additive: an index, never a table rebuild). In LanceDB OSS creation is
        # synchronous; new rows are searchable immediately, and a periodic
        # optimize() folds them into the index for speed at scale.
        if self._fts_ready:
            return
        for index in self._table.list_indices():
            if "content" in getattr(index, "columns", []) and "FTS" in str(
                getattr(index, "index_type", "")
            ).upper():
                self._fts_ready = True
                return
        self._table.create_fts_index("content")
        self._fts_ready = True

    def _where(self, criteria: SearchCriteria) -> str:
        clauses = ["status = 'active'"]
        if criteria.scope == "global":
            clauses.append("scope = 'global'")
        elif criteria.scope != "all":  # 'project' = this project OR global (soft scope)
            if criteria.project is None:
                clauses.append("(project IS NULL OR scope = 'global')")
            else:
                clauses.append(f"(project = {self._quote(criteria.project)} OR scope = 'global')")
        if criteria.type is not None:
            clauses.append(f"type = {self._quote(criteria.type.value)}")
        for tag in criteria.tags:  # ALL tags must be present
            clauses.append(f"array_has(tags, {self._quote(tag)})")
        if criteria.related_files:  # ANY of the files
            ors = " OR ".join(
                f"array_has(related_files, {self._quote(path)})"
                for path in criteria.related_files
            )
            clauses.append(f"({ors})")
        if criteria.created_after is not None:
            clauses.append(f"created_at >= {self._quote(criteria.created_after)}")
        return " AND ".join(clauses)

    def register_duplicate(self, memory_id: str) -> None:
        memory = self._by_id(memory_id)
        if memory is None:
            return
        memory.register_duplicate()
        self._table.update(
            where=f"id = {self._quote(memory_id)}",
            values={
                "duplicate_count": memory.duplicate_count,
                "last_seen_at": memory.last_seen_at,
            },
        )

    def mark_superseded(self, memory_id: str) -> None:
        memory = self._by_id(memory_id)
        if memory is None:
            return
        memory.mark_superseded()
        self._table.update(
            where=f"id = {self._quote(memory_id)}",
            values={"status": memory.status, "updated_at": memory.updated_at},
        )

    def delete(self, ids: list[str]) -> int:
        if self._table is None or not ids:
            return 0
        joined = ", ".join(self._quote(memory_id) for memory_id in ids)
        return self._delete_where(f"id IN ({joined})")

    def delete_by_project(self, project: str) -> int:
        if self._table is None:
            return 0
        return self._delete_where(f"project = {self._quote(project)}")

    def delete_all(self) -> int:
        if self._table is None:
            return 0
        removed = self._table.count_rows()
        self._db.drop_table(_TABLE)
        self._table = None  # recreated on the next add
        return removed

    def list_all(self) -> list[Memory]:
        if self._table is None:
            return []
        return [from_dict(row) for row in self._table.to_arrow().to_pylist()]

    # --- internals ---

    def _ensure_table(self, dim: int):
        if self._table is None:
            self._table = self._db.create_table(_TABLE, schema=self._schema(dim))
        return self._table

    def _first(self, predicate: str) -> Memory | None:
        if self._table is None:
            return None
        rows = self._table.search().where(predicate).limit(1).to_list()
        return from_dict(rows[0]) if rows else None

    def _by_id(self, memory_id: str) -> Memory | None:
        return self._first(f"id = {self._quote(memory_id)}")

    def _delete_where(self, predicate: str) -> int:
        removed = self._table.count_rows(filter=predicate)
        self._table.delete(predicate)
        return removed

    @staticmethod
    def _quote(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    @staticmethod
    def _schema(dim: int):
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("type", pa.string()),
                pa.field("scope", pa.string()),
                pa.field("project", pa.string()),
                pa.field("related_files", pa.list_(pa.string())),
                pa.field("tags", pa.list_(pa.string())),
                pa.field("topic_key", pa.string()),
                pa.field("session_id", pa.string()),
                pa.field("status", pa.string()),
                pa.field("supersedes", pa.string()),
                pa.field("hash", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
                pa.field("last_seen_at", pa.string()),
                pa.field("duplicate_count", pa.int64()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ]
        )
