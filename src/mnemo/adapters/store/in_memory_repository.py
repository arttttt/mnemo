"""In-memory repository: brute-force cosine + optional JSON persistence.

The offline/test backend, and the default until the LanceDB store lands. It
implements MemoryRepositoryPort structurally.
"""
from __future__ import annotations

import json
from pathlib import Path

from mnemo.adapters.store.memory_serializer import from_dict, to_dict
from mnemo.adapters.store.similarity import cosine
from mnemo.application.scored_memory import ScoredMemory
from mnemo.application.search_criteria import SearchCriteria
from mnemo.application.types import Vector
from mnemo.domain.memory import Memory


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

    def find_active_by_topic_key(
        self, topic_key: str, project: str | None
    ) -> Memory | None:
        for memory, _ in self._items:
            if (
                memory.status == "active"
                and memory.topic_key == topic_key
                and memory.project == project
            ):
                return memory
        return None

    def search(
        self, vector: Vector, criteria: SearchCriteria, limit: int
    ) -> list[ScoredMemory]:
        scored = [
            ScoredMemory(memory=memory, score=cosine(vector, stored))
            for memory, stored in self._items
            if criteria.matches(memory)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def register_duplicate(self, memory_id: str) -> None:
        for memory, _ in self._items:
            if memory.id == memory_id:
                memory.register_duplicate()
                break
        self._persist()

    def mark_superseded(self, memory_id: str) -> None:
        for memory, _ in self._items:
            if memory.id == memory_id:
                memory.mark_superseded()
                break
        self._persist()

    def delete(self, ids: list[str]) -> int:
        targets = set(ids)
        return self._remove(lambda memory: memory.id in targets)

    def delete_by_project(self, project: str) -> int:
        return self._remove(lambda memory: memory.project == project)

    def delete_all(self) -> int:
        removed = len(self._items)
        self._items = []
        self._index_by_hash = {}
        self._persist()
        return removed

    def list_all(self) -> list[Memory]:
        return [memory for memory, _ in self._items]

    # --- internals ---

    def _remove(self, should_remove) -> int:
        before = len(self._items)
        self._items = [
            (memory, vector)
            for memory, vector in self._items
            if not should_remove(memory)
        ]
        self._reindex()
        self._persist()
        return before - len(self._items)

    def _reindex(self) -> None:
        self._index_by_hash = {
            memory.hash: index for index, (memory, _) in enumerate(self._items)
        }

    def _persist(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"memory": to_dict(memory), "vector": vector}
            for memory, vector in self._items
        ]
        self._path.write_text(json.dumps(payload))

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        for row in json.loads(self._path.read_text()):
            memory = from_dict(row["memory"])
            self._index_by_hash[memory.hash] = len(self._items)
            self._items.append((memory, list(row["vector"])))
