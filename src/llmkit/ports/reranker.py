"""Port: score documents against a query with a cross-encoder.

Operation-level: one call ranks the whole shortlist; residency is hidden behind it.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    def rank(self, query: str, documents: Sequence[str]) -> list[float]:
        """A relevance score per document (higher = more relevant), aligned to `documents`."""
        ...
