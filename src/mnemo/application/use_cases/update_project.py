"""Update a registered project's description — the only way to set/change it.

Description is what later powers semantic near-match (tier-2) over projects, so it
is editable independently of creation. Updating an unregistered project errors with
near-match candidates, like the gate.
"""
from __future__ import annotations

from mnemo.application.ports.project_repository import ProjectRepository
from mnemo.application.project_gate import UnknownProject, near_match_candidates
from mnemo.domain.project import Project


class UpdateProjectUseCaseImpl:
    def __init__(self, projects: ProjectRepository) -> None:
        self._projects = projects

    def execute(self, name: str, description: str | None) -> Project:
        if not self._projects.exists(name):
            raise UnknownProject(name, near_match_candidates(self._projects, name))
        self._projects.update_description(name, description)
        return self._projects.get(name)
