"""List active memories by filter, newest first — retrieval without a query.

No embedding and no relevance ranking: a filter-only browse (type / tags / scope /
recency) ordered by recency, distinct from the semantic `search`. Calls the store's
`browse` path directly — no query, so no fusion or scoring.
"""
from __future__ import annotations

from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.project_gate import ProjectGate
from mnemo.application.results.browse_result import BrowseResult
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
        limit: int = 10,
    ) -> list[BrowseResult]:
        criteria = SearchCriteria(
            scope=scope,
            project=project,
            type=MemoryType(type) if type else None,
            tags=tuple(tags or ()),
            related_files=tuple(related_files or ()),
            created_after=created_after,
        )
        self._gate.check(criteria.scope, criteria.project)
        memories = self._repository.browse(criteria, limit)
        return [
            BrowseResult(
                id=memory.id,
                type=memory.type.value,
                scope=memory.scope.value,
                project=memory.project,
                content=memory.content,
                related_files=memory.related_files,
                created_at=memory.created_at,
                topic_key=memory.topic_key,
                status=memory.status,
            )
            for memory in memories
        ]
