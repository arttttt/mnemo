"""The scope↔project contract, shared by every scoped operation (search, browse, remember).

`scope='project'` must name a project; `scope='global'`/`'all'` must not — scope is
authoritative. Enforced in one place so the rule can't drift between callers.
"""
from __future__ import annotations


def validate_scope_project(scope: str, project: str | None) -> None:
    # A project scope must name the project; there is no "current project" to infer,
    # so without one the operation would silently target only project-less + global
    # rows (almost always nothing). Reject it with an actionable error instead.
    if scope == "project" and project is None:
        raise ValueError(
            "scope='project' needs a project to scope to, but none was given; "
            "pass the project slug, or use scope='global' (only global memories) "
            "or scope='all' (across every project)"
        )
    # The mirror case: 'all'/'global' ignore project entirely. Accepting one would
    # silently drop the filter and act on the wrong scope, so reject the contradiction.
    if scope in ("global", "all") and project is not None:
        raise ValueError(
            f"project has no effect with scope='{scope}'; drop the project param, "
            f"or use scope='project' to scope to it"
        )
