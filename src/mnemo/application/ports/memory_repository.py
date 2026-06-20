"""Port: persistence and retrieval for memories (the memory aggregate).

The deferred-embedding queue (EmbeddingQueuePort) and the link graph
(LinkGraphPort) are SEPARATE ports — one store implementation realizes all three
facets, but each consumer depends only on the slice it needs.
"""
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
        via FTS5, absent from dense search) until the embedding lands — see deferred
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

    def delete(self, ids: list[str]) -> int: ...

    def delete_by_project(self, project: str) -> int: ...

    def delete_all(self) -> int: ...

    def list_all(self) -> list[Memory]: ...
