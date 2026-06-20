"""Port: persistence and retrieval for memories."""
from __future__ import annotations

from typing import Protocol

from mnemo.application.retrieval import Retrieval
from mnemo.application.scored_memory import ScoredMemory
from mnemo.application.types import Vector
from mnemo.domain.link import Link
from mnemo.domain.memory import Memory


class MemoryRepositoryPort(Protocol):
    def add(self, memory: Memory, vector: Vector | None = None) -> None:
        """Persist a memory. `vector=None` stores it **pending** (lexically searchable
        via FTS5, absent from dense search) until `set_vector` lands — see deferred
        embedding in docs/03-architecture.md."""
        ...

    def supersede(
        self, memory: Memory, link: Link, vector: Vector | None = None
    ) -> None:
        """Atomically persist a supersede: mark `memory.supersedes` (the prior record)
        superseded, persist `memory` (the successor), and write `link` (the supersedes
        edge) — all or nothing. The CALLER owns the relationship (sets
        `memory.supersedes` and builds `link`); the repository only persists it, so a
        crash can never leave the topic_key with no active record, or a successor with
        no provenance edge. `vector=None` stores the successor **pending**, like `add`."""
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

    def set_dimension(self, new_dim: int) -> None:
        """Prepare the store for `new_dim` embeddings, dropping every existing vector to
        pending for re-computation. No-op when the dimension already matches. Used to
        migrate between embedders; content/metadata/links are preserved."""
        ...

    def find_active_by_hash(
        self, content_hash: str, project: str | None
    ) -> Memory | None:
        """The ACTIVE memory whose content hashes to `content_hash` WITHIN `project`,
        or None. Content is unique only within a scope, so the lookup is project-scoped
        (global memories live under the `__global__` sentinel project) and ignores
        superseded rows — mirroring `find_active_by_topic_key`."""
        ...

    def find_active_by_topic_key(
        self, topic_key: str, project: str | None
    ) -> Memory | None: ...

    def retrieve(self, request: Retrieval) -> list[ScoredMemory]:
        """Rank memories for a retrieval request: dense (`request.vector`) and/or
        lexical (`request.text`) legs fused, filtered by `request.criteria`."""
        ...

    def mark_superseded(self, memory_id: str) -> None: ...

    def delete(self, ids: list[str]) -> int: ...

    def delete_by_project(self, project: str) -> int: ...

    def delete_all(self) -> int: ...

    def list_all(self) -> list[Memory]: ...

    def add_link(self, link: Link) -> None: ...

    def links_for(self, memory_id: str) -> list[Link]: ...
