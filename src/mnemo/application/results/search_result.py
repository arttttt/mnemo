"""A single search hit returned to a caller.

No relevance score: the list ORDER is the ranking, and the underlying RRF value is opaque and
misleadable (a consuming agent misreads it as a confidence), so it stays internal — the caller
reads the hit content to judge relevance. See docs/05-mcp-api.md.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    id: str
    type: str
    scope: str
    project: str | None
    content: str
    related_files: list[str]
    created_at: str
