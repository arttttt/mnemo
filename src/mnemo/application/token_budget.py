"""Enforce a token cap on a piece of text — count it (in the injected counter's tokens) and
reject, never truncate, when it's over.

The single home of the write-path length rule, shared by the memory-content cap (per type) and
the project-description cap — so the two can't drift and a "token" means the same thing for both.
Depends on the narrow ``TokenCounter`` port, not the embedder.
"""
from __future__ import annotations

from mnemo.application.ports.token_counter import TokenCounter


class TokenBudget:
    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter

    def ensure_within(
        self, text: str, cap: int, *, subject: str, advice: str, qualifier: str = ""
    ) -> None:
        """Raise ``ValueError`` when ``text`` exceeds ``cap`` tokens. ``subject`` names what was
        measured, ``qualifier`` is an optional clause right after the limit, and ``advice`` tells
        the caller how to fix it — the caller is an LLM, so we reject (never truncate) with enough
        guidance for it to resolve the overflow deliberately."""
        tokens = self._counter.count_tokens(text)
        if tokens > cap:
            raise ValueError(
                f"{subject} is {tokens} tokens, over the {cap}-token limit{qualifier}; {advice}"
            )
