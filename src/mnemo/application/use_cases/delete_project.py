"""Delete a registered project and everything in it.

A single DELETE on the registry; the store's ON DELETE CASCADE foreign keys remove
the project's memories and their links atomically (no app-level orchestration).
Deleting an unregistered project errors with near-match candidates — deletion is
destructive, so never guess at the wrong slug.
"""
from __future__ import annotations

from mnemo.application.ports.project_repository import ProjectRepository
from mnemo.application.project_gate import UnknownProject, near_match_candidates
from mnemo.domain.project import Project


class DeleteProjectUseCaseImpl:
    def __init__(self, projects: ProjectRepository) -> None:
        self._projects = projects

    def execute(self, name: str) -> Project:
        project = self._projects.get(name)
        if project is None:
            raise UnknownProject(name, near_match_candidates(self._projects, name))
        # One DELETE FROM projects; the FK cascade removes its memories and their links.
        self._projects.delete(name)
        return project
