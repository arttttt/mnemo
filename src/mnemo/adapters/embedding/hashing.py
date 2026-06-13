"""Deterministic, dependency-free embedder. NOT semantic — offline/tests/skeleton."""
from __future__ import annotations

import hashlib
import math

from mnemo.application.ports import Vector


class HashEmbedder:
    """Bag-of-tokens hashing embedder.

    Deterministic and free of heavy deps, so the core can be tested offline.
    Lexical only: it captures token overlap, not meaning.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> Vector:
        vector = [0.0] * self._dim
        for token in text.lower().split():
            bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self._dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
