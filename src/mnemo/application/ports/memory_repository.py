"""Port: persistence and retrieval for memories."""
from __future__ import annotations

from typing import Protocol

from mnemo.application.scored_memory import ScoredMemory
from mnemo.application.search_criteria import SearchCriteria
from mnemo.application.types import Vector
from mnemo.domain.link import Link
from mnemo.domain.memory import Memory


class MemoryRepositoryPort(Protocol):
    def add(self, memory: Memory, vector: Vector | None = None) -> None:
        """Persist a memory. `vector=None` stores it **pending** (lexically searchable
        via FTS5, absent from dense search) until `set_vector` lands — see deferred
        embedding in docs/03-architecture.md."""
        ...

    # --- deferred embedding (the DB is the durable embed queue) ---
    def set_vector(self, memory_id: str, vector: Vector) -> None:
        """Attach the embedding to an existing record (upsert; no-op if id is gone)."""
        ...

    def has_vector(self, memory_id: str) -> bool: ...

    def content_for(self, memory_id: str) -> str | None: ...

    def next_unembedded(self, limit: int) -> list[str]:
        """Ids of memories still missing a vector (the pending work-list)."""
        ...

    def pending_count(self) -> int: ...

    def find_by_hash(self, content_hash: str) -> Memory | None: ...

    def find_active_by_topic_key(
        self, topic_key: str, project: str | None
    ) -> Memory | None: ...

    def search(
        self, query: str, vector: Vector, criteria: SearchCriteria, limit: int
    ) -> list[ScoredMemory]: ...

    def register_duplicate(self, memory_id: str) -> None: ...

    def mark_superseded(self, memory_id: str) -> None: ...

    def delete(self, ids: list[str]) -> int: ...

    def delete_by_project(self, project: str) -> int: ...

    def delete_all(self) -> int: ...

    def list_all(self) -> list[Memory]: ...

    def add_link(self, link: Link) -> None: ...

    def links_for(self, memory_id: str) -> list[Link]: ...
