"""The domain eval fixture (п3) is well-formed: it loads, every slice is present, gold/stale keys
resolve to real memories, and rule memories stay terse enough for the live 128-token rule cap."""
from __future__ import annotations

from pathlib import Path

from tools.eval.domain_fixture import SLICES, load_fixture

_FIXTURE = Path(__file__).resolve().parents[2] / "tools" / "eval" / "fixtures" / "domain_v1.json"


def test_fixture_loads_and_validates():
    fixture = load_fixture(_FIXTURE)  # raises on any integrity error
    assert fixture.memories
    assert fixture.questions


def test_every_slice_is_represented():
    fixture = load_fixture(_FIXTURE)
    assert {q.slice for q in fixture.questions} == set(SLICES)


def test_gold_and_stale_keys_resolve_to_memories():
    fixture = load_fixture(_FIXTURE)
    keys = {m.key for m in fixture.memories}
    for q in fixture.questions:
        assert set(q.gold_keys) <= keys
        assert set(q.stale_keys) <= keys


def test_rule_memories_are_terse_enough_for_the_rule_cap():
    # rules ingest as type=rule (128-token cap); ~500 chars is comfortably under that for English,
    # so this catches an over-long rule before the runtime cap would reject it at ingest.
    fixture = load_fixture(_FIXTURE)
    for m in fixture.memories:
        if m.type == "rule":
            assert len(m.content) <= 500, f"{m.key} risks exceeding the 128-token rule cap"
