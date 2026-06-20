"""Retrieve active memories by similarity within a (soft) scope and filters."""
from __future__ import annotations

from mnemo.application.ports.embedder import TextEmbedder
from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.project_gate import ProjectGate
from mnemo.application.results.search_result import SearchResult
from mnemo.application.retrieval import Retrieval
from mnemo.application.search_criteria import SearchCriteria
from mnemo.domain.memory_type import MemoryType


class SearchMemoryUseCaseImpl:
    def __init__(
        self, repository: MemoryRepository, embedder: TextEmbedder, gate: ProjectGate
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._gate = gate

    def execute(
        self,
        *,
        query: str,
        scope: str = "project",
        project: str | None = None,
        type: str | None = None,
        tags: list[str] | None = None,
        related_files: list[str] | None = None,
        created_after: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        criteria = SearchCriteria(
            scope=scope,
            project=project,
            type=MemoryType(type) if type else None,
            tags=tuple(tags or ()),
            related_files=tuple(related_files or ()),
            created_after=created_after,
        )
        self._gate.check(criteria.scope, criteria.project)
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
