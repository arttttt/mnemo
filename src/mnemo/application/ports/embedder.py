"""Port: turns text into a local embedding vector."""
from typing import Protocol

from mnemo.application.types import Vector


class EmbedderPort(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def max_input(self) -> int:
        """Maximum input length the embedder accepts, in tokens (its context window).

        A memory is one vector, so this is the hard upper bound on a memory's size.
        The write use case enforces it (reject, never truncate); the embedder owns the
        number because the window is the model's, not a guess.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Length of `text` in the embedder's own tokens (for the max_input check)."""
        ...

    def encode(self, text: str) -> Vector: ...
