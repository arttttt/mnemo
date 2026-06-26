"""Fuser: merge the two hybrid legs into one RRF-ranked pool and derive the
per-query confidence signals — pure, no SQL."""
from __future__ import annotations

from mnemo.application.fusion.fuser import Fuser
from mnemo.application.fusion.results import ChannelResults
from mnemo.application.scored_memory import ScoredMemory
from mnemo.domain.memory import Memory


def _hit(memory_id: str, score: float) -> ScoredMemory:
    return ScoredMemory(memory=Memory.create(f"content {memory_id}", id=memory_id), score=score)


def test_id_in_both_legs_outranks_one_leg_only():
    channels = ChannelResults(
        dense=(_hit("a", 0.9), _hit("b", 0.8)),
        lexical=(_hit("a", 5.0),),
    )
    pool = Fuser().fuse(channels, limit=10).pool
    assert [m.memory.id for m in pool] == ["a", "b"]  # "a" tops both legs → wins


def test_pool_is_trimmed_to_limit():
    channels = ChannelResults(
        dense=tuple(_hit(c, 1.0 - i * 0.1) for i, c in enumerate("abcde")),
        lexical=(),
    )
    pool = Fuser().fuse(channels, limit=3).pool
    assert [m.memory.id for m in pool] == ["a", "b", "c"]


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
    ).signals
    assert split.agree is False
    assert split.dense_margin == 0.0  # only one dense hit

    empty = Fuser().fuse(ChannelResults(dense=(), lexical=()), limit=5)
    assert empty.pool == ()
    assert empty.signals.dense_top1 == 0.0 and empty.signals.overlap == 0.0
