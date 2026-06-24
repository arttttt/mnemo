"""Loader + integrity validator for the domain eval fixture (the п3 question set).

Pure data: it knows the fixture's shape and self-consistency rules — gold/stale keys resolve to
real memories, slices/categories are known, and a superseded pair is ingested newer-after-older
so the current memory wins. It knows NOTHING about the harness or the metrics; the runner
(domain.py) orchestrates those. Gold is referenced by a stable memory ``key``, never a volatile
store id, so the fixture survives re-ingestion.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SLICES = frozenset({"answerable", "irrelevant", "superseded"})


@dataclass(frozen=True)
class FixtureMemory:
    key: str                          # stable handle for gold references (NOT the store id)
    type: str
    scope: str
    project: str | None
    content: str
    tags: tuple[str, ...]
    related_files: tuple[str, ...]
    topic_key: str | None             # shared across a supersede pair (reuse supersedes the prior)


@dataclass(frozen=True)
class FixtureQuestion:
    id: str
    slice: str                        # answerable | irrelevant | superseded
    category: str
    project: str                      # the project to scope the search to
    question: str
    gold_keys: tuple[str, ...]        # the memory key(s) that answer it (empty for irrelevant)
    stale_keys: tuple[str, ...]       # the outdated version(s) a superseded question must NOT prefer
    answer: str                       # "REFUSE" for the irrelevant slice


@dataclass(frozen=True)
class Fixture:
    version: str
    description: str
    projects: tuple[str, ...]
    memories: tuple[FixtureMemory, ...]
    questions: tuple[FixtureQuestion, ...]


def load_fixture(path: Path) -> Fixture:
    """Parse + validate the fixture; raises ValueError on any integrity violation."""
    raw = json.loads(path.read_text())
    memories = tuple(
        FixtureMemory(
            key=m["key"], type=m["type"], scope=m.get("scope", "project"), project=m.get("project"),
            content=m["content"], tags=tuple(m.get("tags", ())),
            related_files=tuple(m.get("related_files", ())), topic_key=m.get("topic_key"),
        )
        for m in raw["memories"]
    )
    questions = tuple(
        FixtureQuestion(
            id=q["id"], slice=q["slice"], category=q["category"], project=q["project"],
            question=q["question"], gold_keys=tuple(q.get("gold_keys", ())),
            stale_keys=tuple(q.get("stale_keys", ())), answer=q["answer"],
        )
        for q in raw["questions"]
    )
    fixture = Fixture(raw["version"], raw["description"], tuple(raw["projects"]), memories, questions)
    _validate(fixture)
    return fixture


def _validate(fixture: Fixture) -> None:
    keys = {m.key for m in fixture.memories}
    if len(keys) != len(fixture.memories):
        raise ValueError("duplicate memory keys in the fixture")
    ids = [q.id for q in fixture.questions]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question ids in the fixture")
    order = {m.key: i for i, m in enumerate(fixture.memories)}
    for q in fixture.questions:
        if q.slice not in SLICES:
            raise ValueError(f"{q.id}: unknown slice {q.slice!r}")
        unknown = (set(q.gold_keys) | set(q.stale_keys)) - keys
        if unknown:
            raise ValueError(f"{q.id}: references unknown memory keys {sorted(unknown)}")
        if q.slice == "irrelevant" and q.gold_keys:
            raise ValueError(f"{q.id}: an irrelevant question must have no gold_keys")
        if q.slice != "irrelevant" and not q.gold_keys:
            raise ValueError(f"{q.id}: a {q.slice} question needs gold_keys")
        if q.slice == "superseded" and not q.stale_keys:
            raise ValueError(f"{q.id}: a superseded question needs stale_keys")
        # The current (gold) memory must be ingested AFTER its stale version, else the supersede /
        # recency it tests would not hold.
        for stale in q.stale_keys:
            if any(order[stale] > order[gold] for gold in q.gold_keys):
                raise ValueError(f"{q.id}: stale {stale!r} is ingested after its gold")
