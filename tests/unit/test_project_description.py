"""enforce_description_cap — the project-description guard: a missing description is always fine,
and an over-budget one is rejected at the rule-sized 128-token cap with a project-specific message."""
from __future__ import annotations

import pytest

from mnemo.application.project_description import enforce_description_cap
from mnemo.application.token_budget import TokenBudget
from mnemo.domain.constants import PROJECT_DESCRIPTION_MAX_TOKENS


class _Counter:
    def __init__(self, tokens: int) -> None:
        self._tokens = tokens

    def count_tokens(self, text: str) -> int:
        return self._tokens


def test_none_and_empty_descriptions_short_circuit_without_counting():
    # A description that is absent is never measured — even a counter that would report a huge
    # count must not raise, because the guard returns before touching it.
    enforce_description_cap(TokenBudget(_Counter(9999)), None)
    enforce_description_cap(TokenBudget(_Counter(9999)), "")


def test_at_the_cap_passes():
    enforce_description_cap(TokenBudget(_Counter(PROJECT_DESCRIPTION_MAX_TOKENS)), "ok")


def test_over_the_cap_raises_a_project_specific_message():
    with pytest.raises(ValueError, match=r"project description is 200 tokens, over the 128-token limit"):
        enforce_description_cap(TokenBudget(_Counter(200)), "too long")
