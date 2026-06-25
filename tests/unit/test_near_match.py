"""The shared near-match helper used by the project gate and get's not-found errors."""
from mnemo.application.near_match import did_you_mean, near_matches


def test_near_matches_returns_the_closest_candidate_first():
    out = near_matches("auth/modal", ["redis/cache", "auth/model", "auth/jwt"])
    assert out[0] == "auth/model"  # the typo's closest match leads


def test_near_matches_is_empty_when_there_are_no_candidates():
    assert near_matches("x", []) == []


def test_did_you_mean_formats_a_suffix_or_empty():
    assert did_you_mean(["a", "b"]) == " Did you mean: a, b?"
    assert did_you_mean([]) == ""  # nothing to suggest → no suffix
