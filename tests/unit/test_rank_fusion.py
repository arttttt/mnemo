"""Reciprocal-rank fusion: the pure hybrid-search fusion, tested in isolation."""
from mnemo.application.fusion.rank_fusion import reciprocal_rank_fusion


def test_rank_one_in_both_lists_outscores_rank_one_in_one_list():
    # "a" tops both legs; "b" tops only the dense leg → "a" must win.
    scores = reciprocal_rank_fusion([["a", "b"], ["a"]])
    assert scores["a"] > scores["b"]


def test_higher_rank_scores_more_within_a_list():
    scores = reciprocal_rank_fusion([["first", "second", "third"]])
    assert scores["first"] > scores["second"] > scores["third"]


def test_contribution_uses_k_plus_one_based_rank():
    # Single list, default k=60: the top item contributes 1/(60+1).
    scores = reciprocal_rank_fusion([["only"]])
    assert scores["only"] == 1.0 / 61


def test_empty_input_yields_no_scores():
    assert reciprocal_rank_fusion([[], []]) == {}
