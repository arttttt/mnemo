from __future__ import annotations

from typing import Protocol

from mnemo.domain.project import Project


class CreateProjectUseCase(Protocol):
    def execute(self, name: str, description: str | None = None) -> Project: ...
