"""Interface for the search use case."""
from __future__ import annotations

from typing import Protocol

from mnemo.application.results.search_result import SearchResult


class SearchMemoryUseCase(Protocol):
    def execute(
        self,
        *,
        query: str,
        scope: str = "project",
        project: str | None = None,
        type: str | None = None,
        tags: list[str] | None = None,
        related_files: list[str] | None = None,
        recency_days: int | None = None,
        limit: int = 10,
    ) -> list[SearchResult]: ...
