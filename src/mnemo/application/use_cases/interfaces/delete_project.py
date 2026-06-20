from __future__ import annotations

from typing import Protocol

from mnemo.domain.project import Project


class DeleteProjectUseCase(Protocol):
    def execute(self, name: str) -> Project: ...
