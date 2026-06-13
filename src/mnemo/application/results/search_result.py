"""A single search hit returned to a caller."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    id: str
    score: float
    type: str
    scope: str
    project: str | None
    content: str
    related_files: list[str]
    created_at: str
