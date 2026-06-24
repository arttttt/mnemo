"""Interface for the deletion use case."""
from __future__ import annotations

from typing import Protocol

from mnemo.application.results.deletion_result import DeletionResult


class DeleteMemoryUseCase(Protocol):
    def delete(self, ids: list[str], cascade: bool = False) -> DeletionResult: ...

    def purge(self) -> DeletionResult: ...
