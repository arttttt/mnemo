"""Port: turn text into a local embedding vector.

Operation-level: the caller just encodes; loading/unloading is handled by the residency
manager behind this port, so this contract is import-light (no engine imports).
"""
from __future__ import annotations

from typing import Protocol

from llmkit.types import Vector


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def max_input(self) -> int:
        """Maximum input length the model accepts, in its own tokens (its context window)."""
        ...

    def count_tokens(self, text: str) -> int: ...

    def encode(self, text: str) -> Vector: ...
