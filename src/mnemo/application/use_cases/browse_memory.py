"""List active memories by filter, newest first — retrieval without a query.

No embedding and no relevance ranking: a filter-only browse (type / tags / scope /
recency) ordered by recency, distinct from the semantic `search`. Builds a
`Retrieval` with neither text nor vector, so the store takes its browse path.
"""
from __future__ import annotations

from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.project_gate import ProjectGate
from mnemo.application.results.browse_result import BrowseResult
from mnemo.application.retrieval import Retrieval
from mnemo.application.search_criteria import SearchCriteria
from mnemo.domain.memory_type import MemoryType


class BrowseMemoryUseCaseImpl:
    def __init__(self, repository: MemoryRepository, gate: ProjectGate) -> None:
        self._repository = repository
        self._gate = gate

    def execute(
        self,
        *,
        scope: str = "project",
        project: str | None = None,
        type: str | None = None,
        tags: list[str] | None = None,
        related_files: list[str] | None = None,
        created_after: str | None = None,
        status: str = "active",
        limit: int = 10,
    ) -> list[BrowseResult]:
        criteria = SearchCriteria(
            scope=scope,
            project=project,
            type=MemoryType(type) if type else None,
            tags=tuple(tags or ()),
            related_files=tuple(related_files or ()),
            created_after=created_after,
            status=status,
        )
        self._gate.check(criteria.scope, criteria.project)
        scored = self._repository.retrieve(Retrieval(criteria=criteria, limit=limit))
        return [
            BrowseResult(
                id=item.memory.id,
                type=item.memory.type.value,
                scope=item.memory.scope.value,
                project=item.memory.project,
                content=item.memory.content,
                related_files=item.memory.related_files,
                created_at=item.memory.created_at,
                topic_key=item.memory.topic_key,
                status=item.memory.status,
            )
            for item in scored
        ]
