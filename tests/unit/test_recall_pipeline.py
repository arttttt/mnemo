"""The recall pipeline (model-free) — gathers a project's memory and groups it by type.

Recall always takes a query; with no reranker or generator configured it is unused (the
structured grouping is the whole output), but it is still required and validated.
"""
from __future__ import annotations

import pytest

from mnemo.adapters.store.in_memory_repository import InMemoryRepositoryImpl
from mnemo.application.recall.builder import build_recall_pipeline
from mnemo.application.recall.request import RecallRequest
from mnemo.domain.memory import Memory


def _repo_with(*memories: Memory) -> InMemoryRepositoryImpl:
    repo = InMemoryRepositoryImpl()  # pure in-memory, no persistence
    for memory in memories:
        repo.add(memory)  # no vector — the browse path does not need one
    return repo


def test_gathers_a_projects_memory_grouped_by_type():
    repo = _repo_with(
        Memory.create("use jwt", type="decision", project="api"),
        Memory.create("fixed the race", type="debug", project="api"),
        Memory.create("auth adr", type="decision", project="api"),
    )
    bundle = build_recall_pipeline(repo).run(RecallRequest(project="api", query="state"))

    assert bundle.project == "api"
    assert bundle.total == 3
    assert bundle.summary is None  # no generator configured → structured bundle only
    by_type = {section.type: section for section in bundle.sections}
    assert set(by_type) == {"decision", "debug"}
    assert len(by_type["decision"].memories) == 2
    assert len(by_type["debug"].memories) == 1


def test_scopes_to_the_project_but_includes_global_memories():
    repo = _repo_with(
        Memory.create("api decision", type="decision", project="api"),
        Memory.create("other project", type="decision", project="other"),
        Memory.create("a global rule", type="rule", scope="global"),
    )
    bundle = build_recall_pipeline(repo).run(RecallRequest(project="api", query="anything"))

    contents = {m.content for section in bundle.sections for m in section.memories}
    assert contents == {"api decision", "a global rule"}  # 'other' excluded, global kept


def test_limit_caps_the_number_of_gathered_memories():
    repo = _repo_with(
        *[Memory.create(f"note {i}", type="working-notes", project="api") for i in range(5)]
    )
    bundle = build_recall_pipeline(repo).run(RecallRequest(project="api", query="notes", limit=2))
    assert bundle.total == 2


def test_an_empty_query_is_rejected():
    with pytest.raises(ValueError) as exc:
        RecallRequest(project="api", query="   ")  # blank → undirected dump, rejected
    assert "query" in str(exc.value)
