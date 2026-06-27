"""The gate-calibration helpers — separability AUC, the best single-cut, the rank/leg signal
math, and the two query loaders — compute what the report's conclusions rest on, so they're
pinned here (pure functions, no model/store). Corner cases first: empty groups, ties, no shared
ids, a degenerate single-element leg, and the adversarial/no-gold rows the loaders must drop."""
from __future__ import annotations

from types import SimpleNamespace

from tools.eval.gate_calibrate import (
    SIGNAL_KEYS,
    _auc,
    _best_threshold,
    _effect,
    _jaccard_at,
    _overlap_at,
    _softmax_entropy,
    _spearman,
    locomo_queries,
    prod_queries,
    signal_pool,
)


def test_auc_separates_and_is_symmetric():
    assert _auc([0.8, 0.9, 0.85], [0.1, 0.2, 0.15]) == 1.0  # help strictly above hurt
    assert _auc([0.1, 0.2], [0.8, 0.9]) == 0.0              # inverted (still informative)
    assert _auc([0.5, 0.5], [0.5, 0.5]) == 0.5              # all ties → no separation
    assert _auc([], [0.1]) == 0.5 and _auc([0.1], []) == 0.5  # a missing class can't separate


def test_best_threshold_finds_the_clean_cut():
    t, bal, direction = _best_threshold([0.8, 0.9, 0.85], [0.1, 0.2, 0.15])
    assert bal == 1.0 and direction == ">="
    assert 0.2 < t < 0.8  # the cut sits between the two groups
    # When help is the LOW group, the cut flips direction rather than failing.
    _, bal_lo, dir_lo = _best_threshold([0.1, 0.2], [0.8, 0.9])
    assert bal_lo == 1.0 and dir_lo == "<"


def test_effect_labels_including_missing_gold():
    assert _effect(3, 1) == "help"   # bge moved the gold up
    assert _effect(1, 3) == "hurt"   # bge demoted it
    assert _effect(2, 2) == "same"
    assert _effect(None, 1) == "na" and _effect(1, None) == "na"  # gold outside the pool


def test_spearman_and_entropy_corner_cases():
    assert _spearman(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert _spearman(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    assert _spearman(["a"], ["a"]) == 0.0          # <2 shared → undefined → 0
    assert _spearman(["a", "b"], ["x", "y"]) == 0.0  # no shared ids → 0
    assert _softmax_entropy([1, 1, 1, 1]) == 1.0   # flat leg → maximal uncertainty
    assert _softmax_entropy([9, 0, 0, 0]) < 0.2    # one hit dominates → low
    assert _softmax_entropy([1.0]) == 0.0          # single candidate → no entropy


def test_overlap_and_jaccard():
    assert _overlap_at(["a", "b", "c"], ["a", "x", "y"], 2) == 0.5
    assert _jaccard_at(["a", "b"], ["a", "b"], 2) == 1.0
    assert _overlap_at([], [], 0) == 0.0  # k=0 must not divide by zero
    assert _jaccard_at([], [], 5) == 0.0  # empty union → 0, not error


def _leg(ids_scores):
    return tuple(SimpleNamespace(score=s, memory=SimpleNamespace(id=i)) for i, s in ids_scores)


def test_signal_pool_covers_every_key_and_reads_the_legs():
    channels = SimpleNamespace(
        dense=_leg([("a", 0.9), ("b", 0.6), ("c", 0.5)]),
        lexical=_leg([("a", 4.0), ("x", 2.0)]),
    )
    signals = SimpleNamespace(agree=True, overlap=0.3, n_dense=3, n_lexical=2)
    sig = signal_pool(channels, signals, "two short words")
    assert set(sig) == set(SIGNAL_KEYS)             # the report iterates SIGNAL_KEYS — no gaps
    assert sig["dense_top1"] == 0.9
    assert abs(sig["dense_margin"] - 0.3) < 1e-9    # 0.9 - 0.6
    assert sig["agree"] == 1.0
    assert sig["rank_overlap5"] == 0.2              # only "a" shared in the top-5 of each leg
    assert sig["q_words"] == 3.0


def test_signal_pool_survives_an_empty_lexical_leg():
    channels = SimpleNamespace(dense=_leg([("a", 0.7)]), lexical=())
    signals = SimpleNamespace(agree=False, overlap=0.0, n_dense=1, n_lexical=0)
    sig = signal_pool(channels, signals, "q")
    assert sig["bm25_top1"] == 0.0 and sig["dense_top2"] == 0.0  # padded, not IndexError


def test_locomo_queries_drops_adversarial_and_maps_slice():
    data = [{
        "sample_id": "conv-XYZ",
        "qa": [
            {"question": "when?", "evidence": ["d:1", "d:2"], "category": 1},  # multi-hop, kept
            {"question": "trap?", "category": 5},                              # adversarial, no gold
            {"question": "who?", "evidence": ["d:9"], "category": 4},          # single-hop, kept
        ],
    }]
    qs = locomo_queries(data, conversations=None, max_queries=None)
    assert [q.gold for q in qs] == [["d:1", "d:2"], ["d:9"]]   # the no-evidence trap is dropped
    assert qs[0].slice == "multi-hop" and qs[1].slice == "single-hop"
    assert locomo_queries(data, None, max_queries=1) == qs[:1]  # the cap holds


def test_prod_queries_requires_gold():
    rows = [
        {"id": "a1", "slice": "answerable", "project": "p", "question": "q?", "gold_keys": ["k/1"]},
        {"id": "i1", "slice": "irrelevant", "project": "p", "question": "q?", "gold_keys": []},
    ]
    qs = prod_queries(rows)
    assert [q.id for q in qs] == ["a1"]  # the no-gold irrelevant row is unscorable → dropped
