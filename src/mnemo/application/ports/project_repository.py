"""Port: the project registry — projects as first-class, registered entities.

A separate aggregate from the memory store (own table), so a separate interface
and implementation (per rule/no-god-object). Deleting a project removes its
registry row; the DB's FK ON DELETE CASCADE removes the project's memories — the
registry never reaches into the memory tables itself.
"""
from __future__ import annotations

from typing import Protocol

from mnemo.domain.project import Project


class ProjectRepository(Protocol):
    def exists(self, slug: str) -> bool: ...

    def get(self, slug: str) -> Project | None: ...

    def create(self, project: Project) -> None:
        """Insert a new project. The caller ensures it does not already exist."""
        ...

    def update_description(self, slug: str, description: str | None) -> None: ...

    def delete(self, slug: str) -> None:
        """Remove the registry row; the FK cascade removes the project's memories."""
        ...

    def delete_all(self) -> None:
        """Remove every project (a full reset), then re-seed the reserved global
        sentinel so the registry is immediately usable again."""
        ...

    def list_all(self) -> list[Project]:
        """Registered projects, newest first. Excludes the reserved global sentinel."""
        ...
