"""Port: persistence and retrieval for memories (the memory aggregate).

The deferred-embedding queue (EmbeddingQueue) is a SEPARATE port — one store
implementation realizes both facets, but each consumer depends only on the slice
it needs.
"""
from __future__ import annotations

from typing import Protocol

from mnemo.application.results.get_result import ChainEntry
from mnemo.application.retrieval import Retrieval
from mnemo.application.scored_memory import ScoredMemory
from mnemo.application.types import Vector
from mnemo.domain.memory import Memory


class MemoryRepository(Protocol):
    def add(self, memory: Memory, vector: Vector | None = None) -> None:
        """Persist a memory. `vector=None` stores it **pending** (lexically searchable
        via FTS5, absent from dense search) until the embedding lands — see deferred
        embedding in docs/03-architecture.md."""
        ...

    def supersede(self, memory: Memory, vector: Vector | None = None) -> None:
        """Atomically persist a supersede: mark `memory.supersedes` (the prior record)
        superseded and persist `memory` (the successor) — all or nothing. The CALLER
        owns the relationship (sets `memory.supersedes`); the repository only persists
        it, so a crash can never leave the topic_key with no active record. `vector=None`
        stores the successor **pending**, like `add`."""
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

    def find_by_id(self, memory_id: str) -> Memory | None:
        """The memory with this id regardless of status (active or superseded), or None.
        Ids are globally unique, so no scope/project is needed."""
        ...

    def chain(
        self, topic_key: str, project: str | None, *, limit: int, after_id: str | None = None
    ) -> list[ChainEntry]:
        """The supersede lineage of (topic_key, project) — every version sharing the key —
        newest first, each a light ChainEntry. `after_id` is a keyset cursor: return only
        versions OLDER than that id. Bounded by `limit`."""
        ...

    def chain_length(self, topic_key: str, project: str | None) -> int:
        """Total versions in the (topic_key, project) lineage."""
        ...

    def topic_keys(self, project: str | None) -> list[str]:
        """Distinct topic_keys under `project` (any status) — near-match on a get miss."""
        ...

    def retrieve(self, request: Retrieval) -> list[ScoredMemory]:
        """Rank memories for a retrieval request: dense (`request.vector`) and/or
        lexical (`request.text`) legs fused, filtered by `request.criteria`."""
        ...

    def delete(self, ids: list[str], cascade: bool = False) -> int:
        """Delete the given memories. With `cascade=True`, also delete every OLDER member
        each id transitively supersedes (down to the chain root), in one transaction."""
        ...

    def delete_all(self) -> int: ...

    def list_all(self) -> list[Memory]: ...
