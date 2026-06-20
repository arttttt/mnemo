"""The project gate: a scope='project' operation must name a REGISTERED project.

On a miss it raises UnknownProject carrying near-match candidates, so the caller
can pick the real slug (recovery from a typo) or create_project. It needs registry
I/O, so it lives in the application layer BESIDE the pure scope_contract (structural
scope<->project rules), which it does not touch. global/all are scope-authoritative
and exempt — global is not a project.

Near-match is two-tier by design: TIER-1 (here) is difflib over the existing slugs,
on the error path only; TIER-2 (semantic, over project descriptions) is deferred —
see the project-entity/near-match-tier2 note.
"""
from __future__ import annotations

import difflib

from mnemo.application.ports.project_repository import ProjectRepository
from mnemo.domain.scope import Scope

_MAX_CANDIDATES = 5


class UnknownProject(Exception):
    """A scope='project' operation named a project not in the registry."""

    def __init__(self, slug: str, candidates: list[str]) -> None:
        self.slug = slug
        self.candidates = candidates
        suffix = f" Did you mean: {', '.join(candidates)}?" if candidates else ""
        super().__init__(
            f"unknown project {slug!r} — create it with create_project, or use an "
            f"existing slug.{suffix}"
        )


class ProjectGate:
    def __init__(self, projects: ProjectRepository) -> None:
        self._projects = projects

    def check(self, scope: Scope | str, project: str | None) -> None:
        scope_value = scope.value if isinstance(scope, Scope) else scope
        # Only project-scoped operations are gated; global/all are exempt. (The
        # scope<->project structural rules are enforced earlier by scope_contract.)
        if scope_value != Scope.PROJECT.value or project is None:
            return
        if self._projects.exists(project):
            return
        candidates = difflib.get_close_matches(
            project,
            [registered.slug for registered in self._projects.list_all()],
            n=_MAX_CANDIDATES,
            cutoff=0.0,  # top-N closest, no threshold — recovery beats hiding suggestions
        )
        raise UnknownProject(project, candidates)
