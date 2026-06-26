"""Fuse the two hybrid-retrieval legs into one ranked pool and derive the
per-query confidence signals.

Extracted from the store so the ranking lives in the application layer — pure (no
SQL), unit-testable, and shared by the search and recall pipelines. Fusion is
rank-based (RRF), which sidesteps the cosine-vs-BM25 scale mismatch; the signals are
read from the raw legs before RRF flattens them.
"""
from __future__ import annotations

from mnemo.application.fusion.rank_fusion import reciprocal_rank_fusion
from mnemo.application.fusion.results import (
    ChannelResults,
    FusedRetrieval,
    RetrievalSignals,
)
from mnemo.application.scored_memory import ScoredMemory

_OVERLAP_M = 10  # window over which channel agreement (overlap) is measured


class Fuser:
    """Merge ``ChannelResults`` into a ``FusedRetrieval`` (ranked pool + signals)."""

    def fuse(self, channels: ChannelResults, limit: int) -> FusedRetrieval:
        memory_by_id = {hit.memory.id: hit.memory for hit in channels.dense}
        for hit in channels.lexical:
            memory_by_id.setdefault(hit.memory.id, hit.memory)
        dense_ids = [hit.memory.id for hit in channels.dense]
        lexical_ids = [hit.memory.id for hit in channels.lexical]
        fused = reciprocal_rank_fusion([dense_ids, lexical_ids])
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:limit]
        pool = tuple(
            ScoredMemory(memory=memory_by_id[memory_id], score=score)
            for memory_id, score in ranked
        )
        return FusedRetrieval(pool=pool, signals=self._signals(channels))

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
