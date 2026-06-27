"""Create (register) a project — the ONLY path to add a project to the registry.

Writing to an unregistered project is rejected by the gate, so a project must be
created deliberately first. Re-creating an existing slug is a clear error (no
silent update — use update_project to change a description).
"""
from __future__ import annotations

from mnemo.application.ports.project_repository import ProjectRepository
from mnemo.application.ports.token_counter import TokenCounter
from mnemo.application.project_description import enforce_description_cap
from mnemo.application.token_budget import TokenBudget
from mnemo.domain.project import Project


class ProjectAlreadyExists(Exception):
    def __init__(self, slug: str) -> None:
        self.slug = slug
        super().__init__(
            f"project {slug!r} already exists — use update_project to change it"
        )


class CreateProjectUseCaseImpl:
    def __init__(self, projects: ProjectRepository, token_counter: TokenCounter) -> None:
        self._projects = projects
        self._budget = TokenBudget(token_counter)

    def execute(self, name: str, description: str | None = None) -> Project:
        if self._projects.exists(name):
            raise ProjectAlreadyExists(name)
        enforce_description_cap(self._budget, description)
        project = Project.create(name, description)
        self._projects.create(project)
        return project
