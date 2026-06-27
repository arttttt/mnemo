"""Weighted-RRF (the fusion sweep's core): lam=0.5 must reproduce plain RRF's order, lam=1.0 the
dense-primary order (dense first, lexical-only tailing), and a mid lam must let a gold that BOTH
legs rank — but dense ranks low — recover above a dense-only sibling. Also pins the known limit:
a purely lexical-only hit can't be lifted far by weighting alone."""
from __future__ import annotations

from tools.eval.fusion_probe import wrrf_order


def test_lambda_half_is_plain_rrf_order():
    # Both legs rank all three but oppositely → RRF clumps; the dense-first leg breaks ties.
    assert wrrf_order(["a", "b", "c"], ["c", "b", "a"], 0.5) == ["a", "c", "b"]


def test_lambda_one_is_dense_primary_with_lexical_tail():
    assert wrrf_order(["a", "b", "c"], ["c", "b", "a"], 1.0) == ["a", "b", "c"]
    # lexical-only ids append after the whole dense block
    assert wrrf_order(["a", "b"], ["a", "x", "y"], 1.0) == ["a", "b", "x", "y"]


def test_mid_lambda_recovers_a_both_leg_gold_over_a_dense_only_sibling():
    # gold "g": dense rank 2 (low) but BM25 #1; sibling "s": dense #1, absent from lexical.
    # Pure dense (lam=1.0) ranks the sibling over the exact match; weighting the lexical leg back
    # in must lift the exact match above the dense-only sibling — the regression-recovery mechanism.
    dense = ["s", "x", "g"]
    lexical = ["g"]
    assert wrrf_order(dense, lexical, 1.0)[0] == "s"   # dense-primary buries the exact match
    order = wrrf_order(dense, lexical, 0.6)
    assert order.index("g") < order.index("s")          # weighting recovers it


def test_lexical_only_hit_stays_low_even_weighted():
    # A gold present ONLY in the lexical leg (dense never retrieved it) cannot be pulled near the
    # top by weight alone — documents the limit that motivates the splice alternative.
    order = wrrf_order(["a", "b", "c"], ["z", "a"], 0.7)
    assert order[-1] == "z"
