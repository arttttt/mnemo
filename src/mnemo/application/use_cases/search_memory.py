"""Retrieve active memories by similarity within a (soft) scope."""
from __future__ import annotations

from mnemo.application.ports.embedder import EmbedderPort
from mnemo.application.ports.memory_repository import MemoryRepositoryPort
from mnemo.application.results.search_result import SearchResult
from mnemo.application.scoping import scope_predicate


class SearchMemory:
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
        predicate = scope_predicate(scope=scope, project=project, type_filter=type)
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
