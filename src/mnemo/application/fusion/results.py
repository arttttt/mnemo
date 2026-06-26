"""DTOs for hybrid retrieval — the raw per-channel legs, the fused pool, and the
per-query confidence signals computed during fusion.

The store returns the raw legs (``ChannelResults``); the application-layer ``Fuser``
turns them into a ranked ``pool`` plus ``RetrievalSignals``. Keeping fusion and its
signals out of the store lets the ranking policy be inspected and tested without SQL,
and reused by both the search and recall pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass

from mnemo.application.scored_memory import ScoredMemory


@dataclass(frozen=True)
class ChannelResults:
    """The two raw retrieval legs, each best-first with its channel-native score:
    ``dense`` by cosine similarity (higher = better), ``lexical`` by BM25."""

    dense: tuple[ScoredMemory, ...]
    lexical: tuple[ScoredMemory, ...]


@dataclass(frozen=True)
class RetrievalSignals:
    """Per-query retrieval-confidence signals, read from the two legs BEFORE RRF
    flattens them — the input a rerank gate uses to judge whether reranking has
    headroom (a confident runaway vs an ambiguous, channel-split result)."""

    dense_top1: float    # cosine similarity of the best dense hit (0.0 if none)
    dense_margin: float  # dense top-1 minus top-2 similarity (0.0 if fewer than two)
    agree: bool          # the dense and lexical legs share the same rank-1 id
    overlap: float       # |dense ∩ lexical| over the top-M of each, as a fraction of M
    n_dense: int
    n_lexical: int


@dataclass(frozen=True)
class FusedRetrieval:
    """The fused, RRF-ranked ``pool`` (``score`` = the fused score) with its signals."""

    pool: tuple[ScoredMemory, ...]
    signals: RetrievalSignals
