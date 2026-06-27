"""Fuse the two hybrid-retrieval legs into one ranked pool and derive the
per-query confidence signals.

Extracted from the store so the ranking lives in the application layer — pure (no
SQL), unit-testable, and shared by the search and recall pipelines.

Ranking is DENSE-FAVORED weighted reciprocal-rank fusion: each id scores
``w_dense/(k+rank_dense) + w_lexical/(k+rank_lexical)`` with ``w_dense > w_lexical``. This
replaced plain (equal-weight) RRF after two A/Bs on prod + LoCoMo: equal RRF's order clumps
(k=60) and demoted gold that dense similarity ranks correctly, but ranking by dense ALONE
(w_lexical=0) then buried exact-term lookups the BM25 leg had ranked #1. Weighting the dense
leg higher lets it LEAD the order (fixing multi-hop + sibling-crowding) while a strong lexical
match still scores enough to stay near the top. A weight sweep put the knee at w_dense≈0.6 —
multi-hop fixed with no regressions on the fresh snapshot. The signals are still read from the
raw legs, independent of the ranking.
"""
from __future__ import annotations

from mnemo.application.fusion.results import (
    ChannelResults,
    FusedRetrieval,
    RetrievalSignals,
)
from mnemo.application.scored_memory import ScoredMemory
from mnemo.domain.memory import Memory

_OVERLAP_M = 10  # window over which channel agreement (overlap) is measured
RRF_K = 60       # the de-facto RRF constant (dampens the weight of top ranks)
# The dense-leg weight from the sweep ([[bench/weighted-rrf-sweep]]): the knee where dense leads
# the order yet a strong BM25 match is not tail-dumped. w_lexical = 1 - this.
DENSE_WEIGHT = 0.6


class Fuser:
    """Merge ``ChannelResults`` into a ``FusedRetrieval`` (ranked pool + signals).

    ``dense_weight`` is the weighted-RRF balance: 1.0 == rank by dense alone (lexical only as a
    recall tail), 0.5 == plain equal-weight RRF. The default is the tuned sweet spot.
    """

    def __init__(self, dense_weight: float = DENSE_WEIGHT) -> None:
        self._dense_weight = dense_weight

    def fuse(self, channels: ChannelResults, limit: int) -> FusedRetrieval:
        ranked = self._rank(channels)[:limit]
        pool = tuple(ScoredMemory(memory=memory, score=score) for memory, score in ranked)
        return FusedRetrieval(pool=pool, signals=self._signals(channels))

    def _rank(self, channels: ChannelResults) -> list[tuple[Memory, float]]:
        """Weighted reciprocal-rank fusion of the two legs → (memory, fused score) best-first.
        Dense ids come first in the id set, so equal scores keep dense ahead on a stable sort."""
        memory_by_id = {hit.memory.id: hit.memory for hit in channels.dense}
        for hit in channels.lexical:
            memory_by_id.setdefault(hit.memory.id, hit.memory)
        dense_rank = {hit.memory.id: i for i, hit in enumerate(channels.dense)}
        lexical_rank = {hit.memory.id: i for i, hit in enumerate(channels.lexical)}
        w_dense, w_lexical = self._dense_weight, 1.0 - self._dense_weight

        def score(memory_id: str) -> float:
            total = 0.0
            if memory_id in dense_rank:
                total += w_dense / (RRF_K + dense_rank[memory_id])
            if memory_id in lexical_rank:
                total += w_lexical / (RRF_K + lexical_rank[memory_id])
            return total

        scores = {memory_id: score(memory_id) for memory_id in memory_by_id}
        ordered = sorted(memory_by_id.values(), key=lambda m: scores[m.id], reverse=True)
        return [(memory, scores[memory.id]) for memory in ordered]

    @staticmethod
    def _signals(channels: ChannelResults) -> RetrievalSignals:
        dense, lexical = channels.dense, channels.lexical
        dense_top1 = dense[0].score if dense else 0.0
        dense_margin = dense[0].score - dense[1].score if len(dense) >= 2 else 0.0
        agree = bool(dense and lexical and dense[0].memory.id == lexical[0].memory.id)
        top_dense = {hit.memory.id for hit in dense[:_OVERLAP_M]}
        top_lexical = {hit.memory.id for hit in lexical[:_OVERLAP_M]}
        overlap = (
            len(top_dense & top_lexical) / _OVERLAP_M
            if top_dense and top_lexical
            else 0.0
        )
        return RetrievalSignals(
            dense_top1=dense_top1,
            dense_margin=dense_margin,
            agree=agree,
            overlap=overlap,
            n_dense=len(dense),
            n_lexical=len(lexical),
        )
