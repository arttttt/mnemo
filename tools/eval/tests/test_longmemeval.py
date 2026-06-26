"""LongMemEval runner: pure helpers (stratification / NDCG / slug) plus an end-to-end machinery
check on a tiny inline haystack with the hash embedder (no model download)."""
from __future__ import annotations

import math

import pytest

from tools.eval.longmemeval import _slug, ndcg_at_k, stratified


def _q(qid: str, qtype: str) -> dict:
    return {"question_id": qid, "question_type": qtype, "question": "?",
            "haystack_session_ids": [], "haystack_dates": [], "haystack_sessions": [],
            "answer_session_ids": []}


def test_stratified_drops_abstention_and_filters_types():
    data = [_q("a_1", "multi-session"), _q("a_2_abs", "multi-session"),
            _q("b_1", "temporal-reasoning")]
    # _abs is excluded (the benchmark skips abstention for retrieval — no gold location)
    assert {d["question_id"] for d in stratified(data, None, None)} == {"a_1", "b_1"}
    assert [d["question_id"] for d in stratified(data, None, {"temporal-reasoning"})] == ["b_1"]


def test_stratified_balances_across_types():
    data = ([_q(f"u{i}", "single-session-user") for i in range(5)] +
            [_q(f"m{i}", "multi-session") for i in range(5)])
    out = stratified(data, 4, None)
    assert len(out) == 4
    # round-robin → 2 from each type, never 4 from one
    types = [d["question_type"] for d in out]
    assert types.count("single-session-user") == 2
    assert types.count("multi-session") == 2


def test_ndcg_at_k_rewards_higher_rank_and_caps_at_k():
    assert ndcg_at_k(["s1"], {"s1"}, 5) == 1.0
    assert ndcg_at_k([], {"s1"}, 5) == 0.0
    assert ndcg_at_k(["x"], set(), 5) == 0.0                       # no gold → 0, no div-by-zero
    assert ndcg_at_k(["x", "s1"], {"s1"}, 5) == pytest.approx(1 / math.log2(3))  # rank 2 < rank 1
    assert ndcg_at_k(["x", "x", "s1"], {"s1"}, 2) == 0.0           # gold beyond k is not counted


def test_slug_sanitizes_question_ids():
    assert _slug("gpt4_Answer#42") == "gpt4-answer-42"
    assert _slug("") == "q"


def test_runner_retrieves_the_gold_session_end_to_end(tmp_path):
    pytest.importorskip("sqlite_vec")
    from tools.eval import core
    from tools.eval.longmemeval import evaluate

    item = {
        "question_id": "q_capital",
        "question_type": "single-session-user",
        "question": "Which European capital city did I say I love?",
        "haystack_session_ids": ["gold", "distractor"],
        "haystack_dates": ["2026-01-01", "2026-01-02"],
        "haystack_sessions": [
            [{"role": "user", "content": "I love the capital city Paris in Europe."},
             {"role": "assistant", "content": "Paris is a wonderful European capital."}],
            [{"role": "user", "content": "Yesterday I repaired my bicycle tire."},
             {"role": "assistant", "content": "Bicycle maintenance is useful."}],
        ],
        "answer_session_ids": ["gold"],
    }
    container = core.isolated_container(tmp_path, "hash", "")
    report = evaluate(container, [item], [1, 3, 5])

    slice_ = report["by_type"]["single-session-user"]
    assert slice_["n"] == 1
    assert slice_["any_at_k"][5] == 1.0                 # the gold session is retrieved within top-5
    assert report["overall"]["recall_at_k"][5] == 1.0
    assert report["skipped_turns_over_cap"] == 0        # short turns fit the per-type cap
