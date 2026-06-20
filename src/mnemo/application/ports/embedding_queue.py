"""Port: the deferred-embedding work queue (the DB is the queue).

A memory with no vector is a pending job. This is the slice the embedding
schedulers and reindex depend on — distinct from the memory CRUD a use case needs.
One store implementation realizes this alongside MemoryRepositoryPort and
LinkGraphPort, but each consumer depends only on the facet it uses.
"""
from __future__ import annotations

from typing import Protocol

from mnemo.application.types import Vector


class EmbeddingQueuePort(Protocol):
    def set_vector(self, memory_id: str, vector: Vector) -> None:
        """Attach the embedding to an existing record (upsert; no-op if id is gone)."""
        ...

    def has_vector(self, memory_id: str) -> bool: ...

    def content_for(self, memory_id: str) -> str | None: ...

    def next_unembedded(self, limit: int) -> list[str]:
        """Ids of memories still missing a vector (the pending work-list)."""
        ...

    def pending_count(self) -> int: ...

    def set_dimension(self, new_dim: int) -> None:
        """Prepare the store for `new_dim` embeddings, dropping every existing vector to
        pending for re-computation. No-op when the dimension already matches. Used to
        migrate between embedders; content/metadata/links are preserved."""
        ...
