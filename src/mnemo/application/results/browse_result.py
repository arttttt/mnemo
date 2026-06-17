"""A single browse hit — a memory matched by filters, ordered by recency.

Distinct from SearchResult: browse does no relevance ranking, so there is no
`score` (the order itself conveys recency).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BrowseResult:
    id: str
    type: str
    scope: str
    project: str | None
    content: str
    related_files: list[str]
    created_at: str
