"""Decide whether a search result has enough ambiguity to justify a cross-encoder rerank.

Reranking is expensive (a cross-encoder pass over the candidate pool), so it earns its place
only when the cheap hybrid retrieval looks UNCERTAIN — a weak top hit or the two legs
disagreeing on the rank-1 id. A confident, channel-agreeing result is left as-is.

PLACEHOLDER thresholds, pending empirical calibration — the floor below is a first cut, not a
tuned value. Calibrate against the labelled set under the topic ``bench/search-rerank-integration``
(over-fetch a pool, rerank, re-score per slice) before treating these numbers as load-bearing.
"""
from __future__ import annotations

from mnemo.application.fusion.results import RetrievalSignals


class RerankPolicy:
    """Gate the optional rerank stage on the per-query retrieval signals."""

    def __init__(self, *, dense_top1_floor: float = 0.45) -> None:
        # Below this dense top-1 similarity the best hit is weak enough that a rerank has
        # headroom; the value is a PLACEHOLDER (see module docstring) until calibrated.
        self._floor = dense_top1_floor

    def should_rerank(self, signals: RetrievalSignals) -> bool:
        """Rerank when the top dense hit is weak OR the dense and lexical legs disagree on
        rank-1 — the two cheap proxies for an ambiguous result worth a second look."""
        return signals.dense_top1 < self._floor or not signals.agree
