"""Interface for the remember use case."""
from __future__ import annotations

from typing import Protocol

from mnemo.application.results.remember_result import RememberResult
from mnemo.domain.constants import DEFAULT_TYPE
from mnemo.domain.memory_type import MemoryType
from mnemo.domain.scope import Scope


class RememberMemoryUseCase(Protocol):
    def execute(
        self,
        *,
        content: str,
        type: MemoryType | str = DEFAULT_TYPE,
        scope: Scope | str = Scope.PROJECT,
        project: str | None = None,
        related_files: list[str] | None = None,
        tags: list[str] | None = None,
        topic_key: str | None = None,
        session_id: str | None = None,
    ) -> RememberResult: ...
