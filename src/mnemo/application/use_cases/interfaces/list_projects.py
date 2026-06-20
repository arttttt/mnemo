from __future__ import annotations

from typing import Protocol

from mnemo.domain.project import Project


class ListProjectsUseCase(Protocol):
    def execute(self) -> list[Project]: ...
