"""Hard deletion: specific ids, a whole project, or everything."""
from __future__ import annotations

from mnemo.application.ports.memory_repository import MemoryRepositoryPort
from mnemo.application.results.deletion_result import DeletionResult


class DeleteMemory:
    def __init__(self, repository: MemoryRepositoryPort) -> None:
        self._repository = repository

    def delete(self, ids: list[str]) -> DeletionResult:
        return DeletionResult(deleted=self._repository.delete(ids))

    def clear(self, project: str) -> DeletionResult:
        return DeletionResult(deleted=self._repository.delete_by_project(project))

    def purge(self) -> DeletionResult:
        return DeletionResult(deleted=self._repository.delete_all())
