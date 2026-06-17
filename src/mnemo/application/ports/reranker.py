"""Port: a cross-encoder reranker — precise relevance scoring of a shortlist.

It scores each candidate against a query by looking at the pair together, so it runs
only on a shortlist the embedder (or a recency gather) already produced. The model is
heavy to load, so it is used inside a ``session()`` that loads it on entry and frees it
on exit (load → rank → unload) — the same on-demand lifecycle every model-backed stage
uses, keeping it off resident RAM.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol, Sequence


class LoadedReranker(Protocol):
    def rank(self, query: str, documents: Sequence[str]) -> list[float]:
        """A relevance score per document (higher = more relevant), aligned to `documents`."""
        ...


class RerankerPort(Protocol):
    def session(self) -> AbstractContextManager[LoadedReranker]:
        """Load the model for the duration of the context, freeing it on exit."""
        ...
