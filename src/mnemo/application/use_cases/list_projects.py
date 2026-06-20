"""List the registered projects (the reserved __global__ sentinel is excluded)."""
from __future__ import annotations

from mnemo.application.ports.project_repository import ProjectRepository
from mnemo.domain.project import Project


class ListProjectsUseCaseImpl:
    def __init__(self, projects: ProjectRepository) -> None:
        self._projects = projects

    def execute(self) -> list[Project]:
        return self._projects.list_all()
