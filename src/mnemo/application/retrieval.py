"""A retrieval request — what to fetch and how to rank it.

Carries the structured filters (`criteria`), the page size (`limit`), and the
query representation the store ranks by: `text` feeds the lexical (FTS) leg,
`vector` the dense leg. They are the two halves of one hybrid ranking and travel
together: a search carries BOTH; a filter-only browse (recency order) carries
NEITHER. Exactly one without the other is rejected at construction — it would
rank by half the signal and almost always means the caller built the request
wrong (e.g. forgot to embed the query).
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

    def __post_init__(self) -> None:
        # text + vector are the two halves of one hybrid ranking: a search supplies both, a
        # browse neither. Exactly one is never meaningful (the store would rank by half the
        # signal) and signals a caller that built the request wrong — reject it here with a
        # clear, early error instead of letting it crash deep in the store's dense leg.
        if (self.text is None) != (self.vector is None):
            present = "text" if self.text is not None else "vector"
            missing = "vector" if self.vector is None else "text"
            raise ValueError(
                f"a retrieval needs text and vector together (search) or neither (browse); "
                f"got {present} without {missing} — pass both, or neither"
            )
