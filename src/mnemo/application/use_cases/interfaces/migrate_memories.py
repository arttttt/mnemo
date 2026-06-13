"""Interface for the store migration use case."""
from __future__ import annotations

from typing import Protocol

from mnemo.application.results.migration_result import MigrationResult


class MigrateMemoriesUseCase(Protocol):
    def execute(self) -> MigrationResult: ...
