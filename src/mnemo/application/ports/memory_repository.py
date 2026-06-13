"""Port: persistence and retrieval for memories."""
from __future__ import annotations

from typing import Protocol

from mnemo.application.scored_memory import ScoredMemory
from mnemo.application.search_criteria import SearchCriteria
from mnemo.application.types import Vector
from mnemo.domain.memory import Memory


class MemoryRepositoryPort(Protocol):
    def add(self, memory: Memory, vector: Vector) -> None: ...

    def find_by_hash(self, content_hash: str) -> Memory | None: ...

    def find_active_by_topic_key(
        self, topic_key: str, project: str | None
    ) -> Memory | None: ...

    def search(
        self, vector: Vector, criteria: SearchCriteria, limit: int
    ) -> list[ScoredMemory]: ...

    def register_duplicate(self, memory_id: str) -> None: ...

    def mark_superseded(self, memory_id: str) -> None: ...

    def delete(self, ids: list[str]) -> int: ...

    def delete_by_project(self, project: str) -> int: ...

    def delete_all(self) -> int: ...

    def list_all(self) -> list[Memory]: ...
