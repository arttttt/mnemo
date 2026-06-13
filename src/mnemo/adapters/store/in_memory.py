"""In-memory repository: brute-force cosine + optional JSON persistence.

Phase-0 backend. The LanceDB adapter arrives in Phase 1 behind the same port.
(De)serialization is an adapter concern and lives here, keeping the domain pure.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from mnemo.application.ports import (
    MemoryPredicate,
    ScoredMemory,
    Vector,
)
from mnemo.domain.memory import Memory, MemoryType, Scope


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryMemoryRepository:
    def __init__(self, path: str | None = None) -> None:
        self._path = Path(path) if path else None
        self._items: list[tuple[Memory, Vector]] = []
        self._index_by_hash: dict[str, int] = {}
        self._load()

    def add(self, memory: Memory, vector: Vector) -> None:
        self._index_by_hash[memory.hash] = len(self._items)
        self._items.append((memory, list(vector)))
        self._persist()

    def find_by_hash(self, content_hash: str) -> Memory | None:
        index = self._index_by_hash.get(content_hash)
        return self._items[index][0] if index is not None else None

    def search(
        self, vector: Vector, limit: int, predicate: MemoryPredicate | None = None
    ) -> list[ScoredMemory]:
        scored = [
            ScoredMemory(memory=memory, score=_cosine(vector, stored))
            for memory, stored in self._items
            if predicate is None or predicate(memory)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def register_duplicate(self, memory_id: str) -> None:
        for memory, _ in self._items:
            if memory.id == memory_id:
                memory.register_duplicate()
                break
        self._persist()

    def list_all(self) -> list[Memory]:
        return [memory for memory, _ in self._items]

    # --- persistence (adapter concern) ---

    def _persist(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"memory": _to_dict(memory), "vector": vector}
            for memory, vector in self._items
        ]
        self._path.write_text(json.dumps(payload))

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        for row in json.loads(self._path.read_text()):
            memory = _from_dict(row["memory"])
            self._index_by_hash[memory.hash] = len(self._items)
            self._items.append((memory, list(row["vector"])))


def _to_dict(memory: Memory) -> dict:
    return {
        "id": memory.id,
        "content": memory.content,
        "type": memory.type.value,
        "scope": memory.scope.value,
        "project": memory.project,
        "related_files": memory.related_files,
        "tags": memory.tags,
        "importance": memory.importance,
        "topic_key": memory.topic_key,
        "session_id": memory.session_id,
        "status": memory.status,
        "supersedes": memory.supersedes,
        "hash": memory.hash,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "last_seen_at": memory.last_seen_at,
        "duplicate_count": memory.duplicate_count,
    }


def _from_dict(data: dict) -> Memory:
    return Memory(
        id=data["id"],
        content=data["content"],
        type=MemoryType(data["type"]),
        scope=Scope(data["scope"]),
        project=data["project"],
        related_files=data["related_files"],
        tags=data["tags"],
        importance=data["importance"],
        topic_key=data["topic_key"],
        session_id=data["session_id"],
        status=data["status"],
        supersedes=data["supersedes"],
        hash=data["hash"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        last_seen_at=data["last_seen_at"],
        duplicate_count=data["duplicate_count"],
    )
