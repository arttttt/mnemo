"""Port: knows the embedder's token window — its ``max_input`` — and counts tokens.

Shared by TextEmbedder and EmbeddingScheduler so the window-check pair is declared
once and can't drift between them (the scheduler just forwards to the embedder).
Extends ``TokenCounter``: the window adds ``max_input``; counting comes from the counter.
"""
from typing import Protocol

from mnemo.application.ports.token_counter import TokenCounter


class TokenWindow(TokenCounter, Protocol):
    @property
    def max_input(self) -> int:
        """Maximum input length accepted, in tokens (the embedder's context window).

        A memory is one vector, so this is the hard upper bound on a memory's size.
        The write use case enforces it (reject, never truncate); the embedder owns the
        number because the window is the model's, not a guess.
        """
        ...
