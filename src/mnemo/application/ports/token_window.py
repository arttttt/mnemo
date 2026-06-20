"""Port: knows the embedder's token window — its max input and how to count tokens.

Shared by TextEmbedder and EmbeddingScheduler so the window-check pair is declared
once and can't drift between them (the scheduler just forwards to the embedder).
"""
from typing import Protocol


class TokenWindow(Protocol):
    @property
    def max_input(self) -> int:
        """Maximum input length accepted, in tokens (the embedder's context window).

        A memory is one vector, so this is the hard upper bound on a memory's size.
        The write use case enforces it (reject, never truncate); the embedder owns the
        number because the window is the model's, not a guess.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Length of `text` in the embedder's own tokens (for the max_input check)."""
        ...
