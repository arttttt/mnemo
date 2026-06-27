"""The search rerank gate: rerank only an ambiguous result (weak top hit OR channel split)."""
from __future__ import annotations

from mnemo.application.fusion.results import RetrievalSignals
from mnemo.application.search.rerank_policy import RerankPolicy


def _signals(*, dense_top1: float, agree: bool) -> RetrievalSignals:
    return RetrievalSignals(
        dense_top1=dense_top1,
        dense_margin=0.0,
        agree=agree,
        overlap=0.0,
        n_dense=5,
        n_lexical=5,
    )


def test_reranks_when_top_hit_is_weak_even_if_legs_agree():
    policy = RerankPolicy(dense_top1_floor=0.45)
    assert policy.should_rerank(_signals(dense_top1=0.30, agree=True)) is True


def test_reranks_when_legs_disagree_even_if_top_hit_is_strong():
    policy = RerankPolicy(dense_top1_floor=0.45)
    assert policy.should_rerank(_signals(dense_top1=0.90, agree=False)) is True


def test_does_not_rerank_a_confident_agreeing_result():
    policy = RerankPolicy(dense_top1_floor=0.45)
    assert policy.should_rerank(_signals(dense_top1=0.90, agree=True)) is False


def test_floor_is_a_strict_below_comparison():
    policy = RerankPolicy(dense_top1_floor=0.45)
    # exactly at the floor is NOT below it → confident (given the legs agree)
    assert policy.should_rerank(_signals(dense_top1=0.45, agree=True)) is False
