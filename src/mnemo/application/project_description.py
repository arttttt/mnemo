"""The project-description length guard — its cap, subject and advice in one place so
``create_project`` and ``update_project`` reject an over-budget description identically.

Counted like a ``rule`` memory through the shared ``TokenBudget`` (the description is a short,
semantically-matched blurb, not prose). A missing description (``None``/empty) is always fine.
"""
from __future__ import annotations

from mnemo.application.token_budget import TokenBudget
from mnemo.domain.constants import PROJECT_DESCRIPTION_MAX_TOKENS


def enforce_description_cap(budget: TokenBudget, description: str | None) -> None:
    if not description:
        return
    budget.ensure_within(
        description,
        PROJECT_DESCRIPTION_MAX_TOKENS,
        subject="project description",
        advice="tighten the description so it fits within the limit",
    )
