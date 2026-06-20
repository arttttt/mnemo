"""Hard deletion: specific ids, or everything (purge).

A whole project is removed via delete_project — its FK cascade deletes the memories —
so there is no separate per-project clear here.
"""
from __future__ import annotations

from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.ports.project_repository import ProjectRepository
from mnemo.application.results.deletion_result import DeletionResult


class DeleteMemoryUseCaseImpl:
    def __init__(
        self, repository: MemoryRepository, projects: ProjectRepository
    ) -> None:
        self._repository = repository
        self._projects = projects

    def delete(self, ids: list[str]) -> DeletionResult:
        return DeletionResult(deleted=self._repository.delete(ids))

    def purge(self) -> DeletionResult:
        """Full reset: every memory and link, and the project registry too (which
        re-seeds the global sentinel). Returns the count of memories removed."""
        deleted = self._repository.delete_all()
        self._projects.delete_all()
        return DeletionResult(deleted=deleted)
