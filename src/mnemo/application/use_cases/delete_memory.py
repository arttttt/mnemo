"""Hard deletion: specific ids, a whole project, the global memories, or everything."""
from __future__ import annotations

from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.results.deletion_result import DeletionResult
from mnemo.application.scope_contract import validate_scope_project
from mnemo.domain.constants import GLOBAL_PROJECT


class DeleteMemoryUseCaseImpl:
    def __init__(self, repository: MemoryRepository) -> None:
        self._repository = repository

    def delete(self, ids: list[str]) -> DeletionResult:
        return DeletionResult(deleted=self._repository.delete(ids))

    def clear(self, project: str | None = None, *, scope: str = "project") -> DeletionResult:
        """Delete one project's memories (scope='project') or the global memories
        (scope='global'). The scope↔project contract is the same one search/browse
        enforce — globals live under the GLOBAL_PROJECT sentinel, unreachable by a plain
        project slug, so they get their own scope rather than a magic project string."""
        validate_scope_project(scope, project)
        if scope == "all":
            raise ValueError("scope='all' is not allowed for clear; use purge to delete everything")
        target = GLOBAL_PROJECT if scope == "global" else project
        return DeletionResult(deleted=self._repository.delete_by_project(target))

    def purge(self) -> DeletionResult:
        return DeletionResult(deleted=self._repository.delete_all())
