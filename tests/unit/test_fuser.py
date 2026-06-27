"""Fuser: merge the two hybrid legs into one DENSE-FAVORED weighted-RRF pool and derive the
per-query confidence signals — pure, no SQL. The weight has two telling endpoints (1.0 = dense
alone, 0.5 = equal RRF) and a tuned default whose job is to keep dense leading WITHOUT burying a
strong lexical match."""
from __future__ import annotations

from mnemo.application.fusion.fuser import Fuser
from mnemo.application.fusion.results import ChannelResults
from mnemo.application.scored_memory import ScoredMemory
from mnemo.domain.memory import Memory


def _hit(memory_id: str, score: float) -> ScoredMemory:
    return ScoredMemory(memory=Memory.create(f"content {memory_id}", id=memory_id), score=score)


def test_weight_one_ranks_by_dense_alone_with_a_lexical_tail():
    channels = ChannelResults(
        dense=(_hit("a", 0.90), _hit("b", 0.85), _hit("c", 0.80)),
        lexical=(_hit("c", 9.0), _hit("b", 6.0), _hit("a", 3.0)),  # opposite order
    )
    pool = Fuser(dense_weight=1.0).fuse(channels, limit=10).pool
    assert [m.memory.id for m in pool] == ["a", "b", "c"]  # lexical can't perturb the order
    # lexical-only ids tail after the whole dense block
    tail = Fuser(dense_weight=1.0).fuse(
        ChannelResults(dense=(_hit("a", 0.9), _hit("b", 0.8)),
                       lexical=(_hit("a", 9.0), _hit("x", 7.0), _hit("y", 5.0))),
        limit=10,
    ).pool
    assert [m.memory.id for m in tail] == ["a", "b", "x", "y"]


def test_weight_half_is_equal_weight_rrf():
    # Equal weights reproduce plain RRF: with both legs ranking a,b,c oppositely, a and c clump
    # at the top (tie → dense-first) and b sinks.
    channels = ChannelResults(
        dense=(_hit("a", 0.90), _hit("b", 0.85), _hit("c", 0.80)),
        lexical=(_hit("c", 9.0), _hit("b", 6.0), _hit("a", 3.0)),
    )
    pool = Fuser(dense_weight=0.5).fuse(channels, limit=10).pool
    assert [m.memory.id for m in pool] == ["a", "c", "b"]


def test_default_recovers_a_strong_lexical_match_dense_alone_would_bury():
    # "g" is the exact-term gold: dense ranks it #2 behind a higher-cosine sibling "s", but BM25
    # ranks it #1. Dense-alone (1.0) buries g under s; the default weight lifts g back above s.
    channels = ChannelResults(dense=(_hit("s", 0.9), _hit("g", 0.6)), lexical=(_hit("g", 9.0),))
    assert Fuser(dense_weight=1.0).fuse(channels, limit=10).pool[0].memory.id == "s"
    assert Fuser().fuse(channels, limit=10).pool[0].memory.id == "g"


def test_pool_is_trimmed_to_limit():
    channels = ChannelResults(
        dense=tuple(_hit(c, 1.0 - i * 0.1) for i, c in enumerate("abcde")),
        lexical=(),
    )
    pool = Fuser().fuse(channels, limit=3).pool
    assert [m.memory.id for m in pool] == ["a", "b", "c"]


def test_score_is_monotonic_with_position():
    # The pool score is the fused weighted-RRF score, strictly decreasing best→worst.
    pool = Fuser().fuse(
        ChannelResults(dense=(_hit("a", 0.9), _hit("b", 0.1)), lexical=()), limit=10
    ).pool
    assert pool[0].score > pool[1].score


def test_signals_capture_confidence_and_agreement():
    channels = ChannelResults(
        dense=(_hit("a", 0.82), _hit("b", 0.40)),
        lexical=(_hit("a", 7.0), _hit("c", 3.0)),
    )
    signals = Fuser().fuse(channels, limit=10).signals
    assert signals.dense_top1 == 0.82
    assert abs(signals.dense_margin - 0.42) < 1e-9
    assert signals.agree is True  # dense top-1 == lexical top-1 == "a"
    assert signals.n_dense == 2 and signals.n_lexical == 2
    assert abs(signals.overlap - 0.1) < 1e-9  # {a,b} ∩ {a,c} = {a} over top-10


def test_disagreement_single_hit_and_empty_legs():
    split = Fuser().fuse(
        ChannelResults(dense=(_hit("a", 0.5),), lexical=(_hit("b", 2.0),)), limit=10
    )
    assert split.signals.agree is False
    assert split.signals.dense_margin == 0.0  # only one dense hit
    assert [m.memory.id for m in split.pool] == ["a", "b"]  # dense hit leads, lexical-only follows

    empty = Fuser().fuse(ChannelResults(dense=(), lexical=()), limit=5)
    assert empty.pool == ()
    assert empty.signals.dense_top1 == 0.0 and empty.signals.overlap == 0.0
