"""prod_snapshot scoring: with search_cli monkeypatched to canned hits (no real store/subprocess),
the runner scores gold by topic_key per slice, gates scoring on whether a question carries gold, and
treats a no-gold question as the irrelevant (no-refusal) readout rather than a scored miss."""
from __future__ import annotations

from pathlib import Path

from tools.eval import prod_snapshot


def _hit(topic_key: str) -> dict:
    return {"id": topic_key, "type": "decision", "topic_key": topic_key, "content": "..."}


def test_evaluate_scores_gold_by_topic_key_and_gates_on_gold(monkeypatch):
    canned = {
        "q1": [_hit("k1"), _hit("k9"), _hit("k8")],   # gold k1 at rank 1
        "q2": [_hit("k9"), _hit("k8"), _hit("k2")],   # gold k2 at rank 3
        "q3": [_hit("k9"), _hit("k8")],               # irrelevant: nothing answers
        "q4": [_hit("k1"), _hit("k9"), _hit("k2")],   # multi-gold: k1@1, k2@3
    }
    monkeypatch.setattr(prod_snapshot, "search_cli",
                        lambda store, emb, question, project, limit: canned[question])
    questions = [
        {"id": "a1", "project": "p", "slice": "answerable", "question": "q1", "gold_keys": ["k1"]},
        {"id": "a2", "project": "p", "slice": "answerable", "question": "q2", "gold_keys": ["k2"]},
        {"id": "i1", "project": "p", "slice": "irrelevant", "question": "q3", "gold_keys": []},
        {"id": "m1", "project": "p", "slice": "multihop", "question": "q4", "gold_keys": ["k1", "k2"]},
    ]

    report = prod_snapshot.evaluate(Path("unused"), "pplx", questions, [1, 3, 5])
    rows = {r["id"]: r for r in report["per_question"]}

    assert rows["a1"]["gold_rank"] == 0          # rank 1 (0-indexed)
    assert rows["a2"]["gold_rank"] == 2          # rank 3
    assert rows["i1"]["has_gold"] is False       # no gold → not scored, flagged for the irrelevant readout
    assert rows["m1"]["gold_rank"] == 0

    ans = report["slices"]["answerable"]
    assert ans["n"] == 2
    assert ans["recall_at_k"][1] == 0.5          # a1 found@1, a2 not → 0.5
    assert ans["recall_at_k"][3] == 1.0

    mh = report["slices"]["multihop"]
    assert mh["complete_at_k"][1] == 0.0         # both golds needed; only k1 in top-1
    assert mh["complete_at_k"][3] == 1.0         # k1 and k2 both in top-3

    assert "irrelevant" not in report["slices"]  # no-gold slice is never bucketed/scored
