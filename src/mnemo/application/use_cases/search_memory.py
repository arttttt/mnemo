"""Retrieve active memories by similarity within a (soft) scope and filters."""
from __future__ import annotations

from mnemo.application.ports.embedder import EmbedderPort
from mnemo.application.ports.memory_repository import MemoryRepositoryPort
from mnemo.application.results.search_result import SearchResult
from mnemo.application.retrieval import Retrieval
from mnemo.application.search_criteria import SearchCriteria
from mnemo.domain.generators import iso_days_ago
from mnemo.domain.memory_type import MemoryType


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
        tags: list[str] | None = None,
        related_files: list[str] | None = None,
        recency_days: int | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        criteria = SearchCriteria(
            scope=scope,
            project=project,
            type=MemoryType(type) if type else None,
            tags=tuple(tags or ()),
            related_files=tuple(related_files or ()),
            created_after=iso_days_ago(recency_days) if recency_days else None,
        )
        request = Retrieval(
            criteria=criteria,
            limit=limit,
            text=query,
            vector=self._embedder.encode(query),
        )
        scored = self._repository.retrieve(request)
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
