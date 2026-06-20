"""Hard deletion: specific ids, or everything (purge).

A whole project is removed via delete_project — its FK cascade deletes the memories —
so there is no separate per-project clear here.
"""
from __future__ import annotations

from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.results.deletion_result import DeletionResult


class DeleteMemoryUseCaseImpl:
    def __init__(self, repository: MemoryRepository) -> None:
        self._repository = repository

    def delete(self, ids: list[str]) -> DeletionResult:
        return DeletionResult(deleted=self._repository.delete(ids))

    def purge(self) -> DeletionResult:
        return DeletionResult(deleted=self._repository.delete_all())
