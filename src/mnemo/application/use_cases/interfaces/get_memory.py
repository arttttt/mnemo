"""Interface for the get use case."""
from __future__ import annotations

from typing import Protocol

from mnemo.application.results.get_result import GetResult


class GetMemoryUseCase(Protocol):
    def execute(
        self,
        *,
        id: str | None = None,
        topic_key: str | None = None,
        project: str | None = None,
        scope: str = "project",
        chain_limit: int = 10,
        chain_after: str | None = None,
    ) -> GetResult: ...
