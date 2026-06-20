"""In-memory project registry (offline/test double).

The reserved `__global__` sentinel is present so behavior matches the SQLite
backend (exempt from the gate, hidden from listings).
"""
from __future__ import annotations

from mnemo.domain.constants import GLOBAL_PROJECT
from mnemo.domain.generators import now
from mnemo.domain.project import Project


class InMemoryProjectRepositoryImpl:
    def __init__(self) -> None:
        self._projects: dict[str, Project] = {
            GLOBAL_PROJECT: Project(slug=GLOBAL_PROJECT, description=None, created_at=now())
        }

    def exists(self, slug: str) -> bool:
        return slug in self._projects

    def get(self, slug: str) -> Project | None:
        return self._projects.get(slug)

    def create(self, project: Project) -> None:
        self._projects[project.slug] = project

    def update_description(self, slug: str, description: str | None) -> None:
        existing = self._projects.get(slug)
        if existing is not None:
            # New value, not a mutation of the stored entity (repository purity).
            self._projects[slug] = Project(
                slug=existing.slug, description=description, created_at=existing.created_at
            )

    def delete(self, slug: str) -> None:
        self._projects.pop(slug, None)

    def list_all(self) -> list[Project]:
        return [
            project for slug, project in self._projects.items() if slug != GLOBAL_PROJECT
        ]
