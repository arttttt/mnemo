"""In-memory repository: brute-force cosine + optional JSON persistence.

The offline/test backend (the SQLite store is the real one). It implements
MemoryRepositoryPort structurally.
"""
from __future__ import annotations

import json
from pathlib import Path

from mnemo.adapters.store.link_serializer import link_from_dict, link_to_dict
from mnemo.adapters.store.memory_serializer import from_dict, to_dict
from mnemo.adapters.store.similarity import cosine
from mnemo.application.scored_memory import ScoredMemory
from mnemo.application.search_criteria import SearchCriteria
from mnemo.application.types import Vector
from mnemo.domain.link import Link
from mnemo.domain.memory import Memory


class InMemoryMemoryRepository:
    def __init__(self, path: str | None = None) -> None:
        self._path = Path(path) if path else None
        self._items: list[tuple[Memory, Vector]] = []
        self._index_by_hash: dict[str, int] = {}
        self._links: list[Link] = []
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
        self, query: str, vector: Vector, criteria: SearchCriteria, limit: int
    ) -> list[ScoredMemory]:
        # Offline/test backend: it approximates hybrid with cosine over the (already
        # lexical) hash-embedding, so the raw `query` text is not needed here. The
        # real dense+lexical fusion lives in the SQLite backend.
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
        self._links = []
        self._persist()
        return removed

    def list_all(self) -> list[Memory]:
        return [memory for memory, _ in self._items]

    def add_link(self, link: Link) -> None:
        self._links.append(link)
        self._persist()

    def links_for(self, memory_id: str) -> list[Link]:
        return [
            link
            for link in self._links
            if memory_id in (link.source_id, link.target_id)
        ]

    # --- internals ---

    def _remove(self, should_remove) -> int:
        removed_ids = {memory.id for memory, _ in self._items if should_remove(memory)}
        self._items = [
            (memory, vector)
            for memory, vector in self._items
            if not should_remove(memory)
        ]
        # Drop edges that would dangle once their endpoint is gone.
        self._links = [
            link
            for link in self._links
            if link.source_id not in removed_ids and link.target_id not in removed_ids
        ]
        self._reindex()
        self._persist()
        return len(removed_ids)

    def _reindex(self) -> None:
        self._index_by_hash = {
            memory.hash: index for index, (memory, _) in enumerate(self._items)
        }

    def _persist(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "memories": [
                {"memory": to_dict(memory), "vector": vector}
                for memory, vector in self._items
            ],
            "links": [link_to_dict(link) for link in self._links],
        }
        self._path.write_text(json.dumps(payload))

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        raw = json.loads(self._path.read_text())
        # Back-compat: the pre-links format was a bare list of memory rows.
        rows = raw["memories"] if isinstance(raw, dict) else raw
        for row in rows:
            memory = from_dict(row["memory"])
            self._index_by_hash[memory.hash] = len(self._items)
            self._items.append((memory, list(row["vector"])))
        if isinstance(raw, dict):
            self._links = [link_from_dict(link) for link in raw.get("links", [])]
