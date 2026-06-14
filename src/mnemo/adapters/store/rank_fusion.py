"""Reciprocal-rank fusion of ranked id lists.

A pure function so the hybrid-search fusion is testable in isolation, away from
SQL. Each list is in best-first order; an id's contribution is ``1/(k+rank)``,
summed across the lists it appears in. Rank-based fusion sidesteps the scale
mismatch between cosine distance and BM25 — only the ordering matters.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

RRF_K = 60  # the de-facto standard constant (dampens the weight of top ranks)


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Sequence[str]], k: int = RRF_K
) -> dict[str, float]:
    """Fuse best-first id lists into ``id -> fused score`` (higher = better)."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, identifier in enumerate(ranked):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + rank + 1)
    return scores
