"""LanceDB-backed repository: real ANN + persistence behind MemoryRepositoryPort.

The scaling store for the memory layer. It implements MemoryRepositoryPort
structurally; the in-memory store stays the offline/test default. The table is
created lazily on the first write, once the embedding dimension is known.
"""
from __future__ import annotations

from mnemo.adapters.store.memory_serializer import from_dict, to_dict
from mnemo.application.scored_memory import ScoredMemory
from mnemo.application.types import MemoryPredicate, Vector
from mnemo.domain.memory import Memory

_TABLE = "memories"
# Vector search returns nearest-first; the (Python) scope/type predicate is applied
# afterwards, so over-fetch candidates to still fill `limit`. Pushing filters into
# the query itself is a separate, later concern.
_CANDIDATE_FACTOR = 10
_MIN_CANDIDATES = 100


class LanceDbMemoryRepository:
    def __init__(self, uri: str) -> None:
        import lancedb  # heavy, optional dependency — import lazily

        self._db = lancedb.connect(uri)
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
        self, vector: Vector, limit: int, predicate: MemoryPredicate | None = None
    ) -> list[ScoredMemory]:
        if self._table is None:
            return []
        candidates = max(limit * _CANDIDATE_FACTOR, _MIN_CANDIDATES)
        rows = (
            self._table.search(list(vector))
            .metric("cosine")
            .limit(candidates)
            .to_list()
        )
        results: list[ScoredMemory] = []
        for row in rows:
            memory = from_dict(row)
            if predicate is not None and not predicate(memory):
                continue
            # LanceDB cosine distance = 1 - cosine similarity; restore similarity.
            results.append(ScoredMemory(memory=memory, score=1.0 - row["_distance"]))
            if len(results) >= limit:
                break
        return results

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
