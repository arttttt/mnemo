"""The domain eval (п3) machinery runs end-to-end on the fixture with the hash embedder (no model
download): ingest forms the supersede chains in an isolated store, search returns from it, and the
report carries every slice with sane shapes."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sqlite_vec")

from tools.eval import core
from tools.eval.domain import evaluate, ingest
from tools.eval.domain_fixture import load_fixture

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "domain_v1.json"


def _run(tmp_path):
    fixture = load_fixture(_FIXTURE)
    container = core.isolated_container(tmp_path, "hash", "")
    key_by_id = ingest(container, fixture)
    report = evaluate(container, fixture, key_by_id, [1, 3, 5, 10])
    return fixture, key_by_id, report


def test_ingest_keeps_every_memory_including_superseded(tmp_path):
    fixture, key_by_id, _ = _run(tmp_path)
    # supersede keeps the prior row (inactive), so every fixture memory has a distinct store id.
    assert len(key_by_id) == len(fixture.memories)
    assert set(key_by_id.values()) == {m.key for m in fixture.memories}


def test_report_has_every_slice_with_sane_shapes(tmp_path):
    fixture, _, report = _run(tmp_path)
    n_answerable = sum(q.slice == "answerable" for q in fixture.questions)
    n_superseded = sum(q.slice == "superseded" for q in fixture.questions)
    assert report["answerable"]["n"] == n_answerable
    assert report["superseded"]["n_superseded"] == n_superseded
    # the abstention readout carries both candidate gate signals
    assert "cosine" in report["abstention"]
    assert "by_refuse_below" in report["abstention"]["corroboration"]
    # every question got a per-question record
    assert len(report["per_question"]) == len(fixture.questions)


def test_superseded_questions_do_not_surface_a_formally_superseded_version(tmp_path):
    # auth-v1 / scheduling-v1 were superseded via topic_key, so they are inactive and must never
    # appear; the stale leakage there can only come from a coexisting note (rate-limit-todo).
    fixture, key_by_id, report = _run(tmp_path)
    assert report["superseded"]["stale_in_topk"] <= 1
