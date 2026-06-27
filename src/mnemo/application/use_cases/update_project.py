"""Update a registered project's description — the only way to set/change it.

Description is what later powers semantic near-match (tier-2) over projects, so it
is editable independently of creation. Updating an unregistered project errors with
near-match candidates, like the gate.
"""
from __future__ import annotations

from mnemo.application.ports.project_repository import ProjectRepository
from mnemo.application.ports.token_counter import TokenCounter
from mnemo.application.project_description import enforce_description_cap
from mnemo.application.project_gate import UnknownProject, near_match_candidates
from mnemo.application.token_budget import TokenBudget
from mnemo.domain.project import Project


class UpdateProjectUseCaseImpl:
    def __init__(self, projects: ProjectRepository, token_counter: TokenCounter) -> None:
        self._projects = projects
        self._budget = TokenBudget(token_counter)

    def execute(self, name: str, description: str | None) -> Project:
        if not self._projects.exists(name):
            raise UnknownProject(name, near_match_candidates(self._projects, name))
        enforce_description_cap(self._budget, description)
        self._projects.update_description(name, description)
        return self._projects.get(name)
