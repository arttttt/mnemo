"""A retrieval request — what to fetch and how to rank it.

Carries the structured filters (`criteria`), the page size (`limit`), and the
query representation the store ranks by: `text` feeds the lexical (FTS) leg,
`vector` the dense leg. Today every request carries both (semantic search).
Either may be absent — a request with neither `text` nor `vector` is a
filter-only browse ordered by recency, wired in a later step.
"""
from __future__ import annotations

from dataclasses import dataclass

from mnemo.application.search_criteria import SearchCriteria
from mnemo.application.types import Vector


@dataclass(frozen=True)
class Retrieval:
    criteria: SearchCriteria
    limit: int = 10
    text: str | None = None       # the lexical (FTS) leg ranks by this
    vector: Vector | None = None   # the dense leg ranks by this
