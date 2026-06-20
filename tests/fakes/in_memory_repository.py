"""In-memory repository: brute-force cosine + optional JSON persistence.

The offline/test backend (the SQLite store is the real one). It implements
MemoryRepository structurally.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from mnemo.adapters.store.link_serializer import link_from_dict, link_to_dict
from mnemo.adapters.store.memory_serializer import from_dict, to_dict
from mnemo.adapters.store.similarity import cosine
from mnemo.application.retrieval import Retrieval
from mnemo.application.scored_memory import ScoredMemory
from mnemo.application.types import Vector
from mnemo.domain.link import Link
from mnemo.domain.memory import Memory


class InMemoryRepositoryImpl:
    def __init__(self, path: str | None = None) -> None:
        self._path = Path(path) if path else None
        self._items: list[tuple[Memory, Vector | None]] = []
        self._links: list[Link] = []
        self._load()

    def add(self, memory: Memory, vector: Vector | None = None) -> None:
        self._items.append((memory, list(vector) if vector is not None else None))
        self._persist()

    def supersede(
        self, memory: Memory, link: Link, vector: Vector | None = None
    ) -> None:
        """All-or-nothing supersede, mirroring the SQLite backend's transaction. The
        real atomicity guarantee belongs to the SQLite store; here a deep-copied
        snapshot is restored on any failure so the contract test sees the same
        rollback on both backends. The caller owns the relationship (sets
        `memory.supersedes`, builds `link`); this only persists it."""
        items_before = list(self._items)
        links_before = list(self._links)
        try:
            # Replace the prior with a superseded COPY — the stored entity is never
            # mutated, so a shallow snapshot is enough to roll back.
            self._items = [
                (self._superseded(stored), v) if stored.id == memory.supersedes else (stored, v)
                for stored, v in self._items
            ]
            self._items.append((memory, list(vector) if vector is not None else None))
            self._insert_link(link)
            self._persist()
        except BaseException:
            self._items = items_before
            self._links = links_before
            self._persist()
            raise

    @staticmethod
    def _superseded(memory: Memory) -> Memory:
        """A superseded COPY of `memory`: reuse the domain transition without mutating
        the stored entity (the value-store analogue of an UPDATE)."""
        replaced = copy.deepcopy(memory)
        replaced.mark_superseded()
        return replaced

    def set_vector(self, memory_id: str, vector: Vector) -> None:
        for index, (memory, _) in enumerate(self._items):
            if memory.id == memory_id:
                self._items[index] = (memory, list(vector))
                self._persist()
                return

    def has_vector(self, memory_id: str) -> bool:
        return any(
            memory.id == memory_id and stored is not None
            for memory, stored in self._items
        )

    def content_for(self, memory_id: str) -> str | None:
        for memory, _ in self._items:
            if memory.id == memory_id:
                return memory.content
        return None

    def next_unembedded(self, limit: int) -> list[str]:
        return [m.id for m, vec in self._items if vec is None][:limit]

    def pending_count(self) -> int:
        return sum(1 for _, vec in self._items if vec is None)

    def set_dimension(self, new_dim: int) -> None:
        current = next((len(vec) for _, vec in self._items if vec is not None), None)
        if current is None or current == new_dim:
            return  # empty or already at this dimension — nothing to migrate
        self._items = [(memory, None) for memory, _ in self._items]  # drop all to pending
        self._persist()

    def find_active_by_hash(
        self, content_hash: str, project: str | None
    ) -> Memory | None:
        for memory, _ in self._items:
            if (
                memory.status == "active"
                and memory.hash == content_hash
                and memory.project == project
            ):
                return memory
        return None

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

    def retrieve(self, request: Retrieval) -> list[ScoredMemory]:
        # No query text and no vector → filter-only browse, newest first. Pending
        # (un-embedded) memories are included (no vector needed to rank them).
        if request.text is None and request.vector is None:
            matched = [
                memory for memory, _ in self._items if request.criteria.matches(memory)
            ]
            matched.sort(key=lambda memory: memory.created_at, reverse=True)
            return [ScoredMemory(memory=memory, score=0.0) for memory in matched[: request.limit]]

        # Offline/test backend: it approximates hybrid with cosine over the (already
        # lexical) hash-embedding, so the raw `request.text` is not needed here. The
        # real dense+lexical fusion lives in the SQLite backend.
        # Pending (un-embedded) memories have no vector → absent from this dense
        # backend. (The SQLite store still surfaces them via the FTS5 lexical leg.)
        scored = [
            ScoredMemory(memory=memory, score=cosine(request.vector, stored))
            for memory, stored in self._items
            if stored is not None and request.criteria.matches(memory)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: request.limit]

    def delete(self, ids: list[str]) -> int:
        targets = set(ids)
        return self._remove(lambda memory: memory.id in targets)

    def delete_by_project(self, project: str) -> int:
        return self._remove(lambda memory: memory.project == project)

    def delete_all(self) -> int:
        removed = len(self._items)
        self._items = []
        self._links = []
        self._persist()
        return removed

    def list_all(self) -> list[Memory]:
        return [memory for memory, _ in self._items]

    def add_link(self, link: Link) -> None:
        self._insert_link(link)
        self._persist()

    def _insert_link(self, link: Link) -> None:
        self._links.append(link)

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
        self._persist()
        return len(removed_ids)

    def _persist(self) -> None:
        # NOTE: this is the offline/test backend (the SQLite store is the production one,
        # crash-safe via its WAL). The write below is a plain truncate-then-write — NOT
        # atomic and with no recovery — which is acceptable HERE precisely because no real
        # data lives in this backend. Do not promote it to production without a temp-file
        # + os.replace; the persistence guarantees belong to the SQLite store.
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
            stored = row["vector"]
            self._items.append((memory, list(stored) if stored is not None else None))
        if isinstance(raw, dict):
            self._links = [link_from_dict(link) for link in raw.get("links", [])]
