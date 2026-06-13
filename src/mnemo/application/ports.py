"""Application ports (interfaces). Depend only on the domain (NFR-21).

Adapters implement these; use cases depend on them. Dependency Inversion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from mnemo.domain.memory import Memory

Vector = list[float]
MemoryPredicate = Callable[[Memory], bool]


@dataclass(frozen=True)
class ScoredMemory:
    memory: Memory
    score: float


class EmbedderPort(Protocol):
    """Turns text into a local embedding vector."""

    @property
    def dim(self) -> int: ...

    def encode(self, text: str) -> Vector: ...


class MemoryRepositoryPort(Protocol):
    """Persistence and similarity retrieval for memories."""

    def add(self, memory: Memory, vector: Vector) -> None: ...

    def find_by_hash(self, content_hash: str) -> Memory | None: ...

    def search(
        self, vector: Vector, limit: int, predicate: MemoryPredicate | None = None
    ) -> list[ScoredMemory]: ...

    def register_duplicate(self, memory_id: str) -> None: ...

    def list_all(self) -> list[Memory]: ...
