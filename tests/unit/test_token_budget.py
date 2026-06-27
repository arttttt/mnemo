"""TokenBudget — the shared write-path cap guard: count via the injected counter, reject (never
truncate) over the cap, with a message assembled from subject / qualifier / advice. Pinned so the
memory-content and project-description caps that both depend on it can't drift."""
from __future__ import annotations

import pytest

from mnemo.application.token_budget import TokenBudget


class _Counter:
    """A stand-in TokenCounter returning a fixed count, so the boundary is exact (no real tokenizer)."""

    def __init__(self, tokens: int) -> None:
        self._tokens = tokens

    def count_tokens(self, text: str) -> int:
        return self._tokens


def _budget(tokens: int) -> TokenBudget:
    return TokenBudget(_Counter(tokens))


def test_within_and_exactly_at_the_cap_pass():
    _budget(127).ensure_within("x", 128, subject="s", advice="fix it")  # under the cap
    _budget(128).ensure_within("x", 128, subject="s", advice="fix it")  # exactly at the cap is fine


def test_over_the_cap_raises_with_count_cap_and_advice():
    with pytest.raises(ValueError) as exc:
        _budget(200).ensure_within("x", 128, subject="content", advice="condense it")
    msg = str(exc.value)
    assert "content is 200 tokens, over the 128-token limit" in msg
    assert msg.endswith("; condense it")  # the advice is the tail


def test_qualifier_lands_right_after_the_limit():
    with pytest.raises(ValueError) as exc:
        _budget(200).ensure_within(
            "x", 128, subject="content", qualifier=" for a 'rule' memory", advice="a"
        )
    assert "over the 128-token limit for a 'rule' memory; a" in str(exc.value)
