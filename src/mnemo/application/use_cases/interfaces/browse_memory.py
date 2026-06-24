"""Interface for the browse use case."""
from __future__ import annotations

from typing import Protocol

from mnemo.application.results.browse_result import BrowseResult


class BrowseMemoryUseCase(Protocol):
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
    ) -> list[BrowseResult]: ...
