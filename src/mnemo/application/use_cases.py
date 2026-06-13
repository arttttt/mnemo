"""Application use cases — orchestrate the domain via ports. No framework deps."""
from __future__ import annotations

from dataclasses import dataclass

from mnemo.application.ports import (
    EmbedderPort,
    MemoryPredicate,
    MemoryRepositoryPort,
)
from mnemo.domain.memory import DEFAULT_TYPE, Memory, MemoryType, Scope

NEAR_DUPLICATE_THRESHOLD = 0.95


@dataclass(frozen=True)
class RememberResult:
    id: str
    dedup: str | None = None  # None | "exact" | "near"
    score: float | None = None


@dataclass(frozen=True)
class SearchResult:
    id: str
    score: float
    type: str
    scope: str
    project: str | None
    content: str
    related_files: list[str]
    created_at: str


class RememberMemory:
    """Store a memory. No LLM on this path: embed -> dedup -> persist."""

    def __init__(
        self,
        repository: MemoryRepositoryPort,
        embedder: EmbedderPort,
        near_threshold: float = NEAR_DUPLICATE_THRESHOLD,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._near_threshold = near_threshold

    def execute(
        self,
        *,
        content: str,
        type: MemoryType | str = DEFAULT_TYPE,
        scope: Scope | str = Scope.PROJECT,
        project: str | None = None,
        related_files: list[str] | None = None,
        tags: list[str] | None = None,
        importance: float = 0.5,
        topic_key: str | None = None,
        session_id: str | None = None,
    ) -> RememberResult:
        memory = Memory.create(
            content=content,
            type=type,
            scope=scope,
            project=project,
            related_files=related_files,
            tags=tags,
            importance=importance,
            topic_key=topic_key,
            session_id=session_id,
        )

        exact = self._repository.find_by_hash(memory.hash)
        if exact is not None:
            self._repository.register_duplicate(exact.id)
            return RememberResult(id=exact.id, dedup="exact")

        vector = self._embedder.encode(memory.content)
        nearest = self._repository.search(
            vector, limit=1, predicate=_same_bucket(memory)
        )
        if nearest and nearest[0].score >= self._near_threshold:
            return RememberResult(
                id=nearest[0].memory.id,
                dedup="near",
                score=round(nearest[0].score, 4),
            )

        self._repository.add(memory, vector)
        return RememberResult(id=memory.id)


class SearchMemory:
    """Retrieve memories by semantic similarity within a (soft) scope."""

    def __init__(
        self, repository: MemoryRepositoryPort, embedder: EmbedderPort
    ) -> None:
        self._repository = repository
        self._embedder = embedder

    def execute(
        self,
        *,
        query: str,
        scope: str = "project",
        project: str | None = None,
        type: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        vector = self._embedder.encode(query)
        predicate = _scope_predicate(scope=scope, project=project, type_filter=type)
        scored = self._repository.search(vector, limit=limit, predicate=predicate)
        return [
            SearchResult(
                id=item.memory.id,
                score=round(item.score, 4),
                type=item.memory.type.value,
                scope=item.memory.scope.value,
                project=item.memory.project,
                content=item.memory.content,
                related_files=item.memory.related_files,
                created_at=item.memory.created_at,
            )
            for item in scored
        ]


def _same_bucket(memory: Memory) -> MemoryPredicate:
    """Near-dup candidates: active memories of the same type+scope+project."""

    def predicate(other: Memory) -> bool:
        return (
            other.status == "active"
            and other.type == memory.type
            and other.scope == memory.scope
            and other.project == memory.project
        )

    return predicate


def _scope_predicate(
    *, scope: str, project: str | None, type_filter: str | None
) -> MemoryPredicate:
    """Soft scoping: 'project' = this project OR global; 'all' = cross-project."""
    wanted_type = MemoryType(type_filter) if type_filter else None

    def predicate(memory: Memory) -> bool:
        if memory.status != "active":
            return False
        if wanted_type is not None and memory.type != wanted_type:
            return False
        if scope == "all":
            return True
        if scope == "global":
            return memory.scope is Scope.GLOBAL
        return memory.project == project or memory.scope is Scope.GLOBAL

    return predicate
