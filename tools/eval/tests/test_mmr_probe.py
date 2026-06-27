"""The MMR selection — the coverage mechanism the probe measures — is pinned here: it must demote
a near-duplicate in favour of a diverse-but-slightly-less-relevant candidate (the whole point), and
degrade to a plain relevance sort when lambda=1.0. Corner cases: k cap, empty pool, single item."""
from __future__ import annotations

from tools.eval.mmr_probe import _rank_of, mmr_order


# Three candidates: 0 and 1 are near-duplicates (both highly relevant), 2 is diverse.
_REL = [0.98, 0.97, 0.74]
_VECS = [[1.0, 0.0, 0.0], [0.98, 0.05, 0.0], [0.6, 0.0, 0.8]]


def test_lambda_one_is_a_plain_relevance_sort():
    assert mmr_order(_REL, _VECS, 1.0, 3) == [0, 1, 2]  # no diversity term → by relevance


def test_diversity_demotes_the_near_duplicate():
    # The diverse candidate (2) is pulled above the near-duplicate (1) despite lower relevance —
    # this is exactly the multi-hop coverage behaviour a pure reranker cannot produce.
    assert mmr_order(_REL, _VECS, 0.5, 3) == [0, 2, 1]
    assert mmr_order(_REL, _VECS, 0.3, 3)[1] == 2  # stronger diversity keeps 2 at rank 2


def test_k_caps_the_greedy_selection_then_appends_rest_in_order():
    # Only k picks are MMR-selected; the tail keeps the incoming (RRF) order so ranks stay defined.
    order = mmr_order(_REL, _VECS, 0.5, 1)
    assert order[0] == 0 and sorted(order) == [0, 1, 2] and order[1:] == [1, 2]


def test_empty_and_singleton_pools():
    assert mmr_order([], [], 0.7, 5) == []
    assert mmr_order([0.9], [[1.0, 0.0]], 0.7, 5) == [0]


def test_rank_of_finds_first_keyset_carrying_the_gold():
    order_keys = [{"x"}, {"gold", "y"}, set()]
    assert _rank_of(order_keys, "gold") == 1
    assert _rank_of(order_keys, "missing") is None
