"""Deterministic, dependency-free embedder. NOT semantic — offline/tests/skeleton."""
from __future__ import annotations

import hashlib
import math

from mnemo.application.types import Vector


class HashEmbedder:
    """Bag-of-tokens hashing embedder.

    Deterministic and free of heavy deps, so the core can be tested offline.
    Lexical only: it captures token overlap, not meaning.
    """

    def __init__(self, dim: int = 256, max_input: int = 1_000_000) -> None:
        self._dim = dim
        # The offline hash embedder has no meaningful window; default is effectively
        # unlimited. Tests pass a small value to exercise the over-window reject.
        self._max_input = max_input

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_input(self) -> int:
        return self._max_input

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def encode(self, text: str) -> Vector:
        vector = [0.0] * self._dim
        for token in text.lower().split():
            bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self._dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
