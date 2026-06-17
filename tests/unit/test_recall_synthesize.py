"""Synthesized recall — a generator turns the grouped bundle into a query-focused summary."""
from __future__ import annotations

from contextlib import contextmanager

from mnemo.adapters.store.in_memory_repository import InMemoryMemoryRepository
from mnemo.application.recall.builder import build_recall_pipeline
from mnemo.application.recall.request import RecallRequest
from mnemo.domain.memory import Memory


class _EchoGenerator:
    """A deterministic stand-in: asserts the prompt was built, returns a fixed summary."""

    @contextmanager
    def session(self):
        yield self

    def generate(self, prompt, *, max_tokens):
        assert "auth jwt rotation" in prompt            # the gathered memory reached the prompt
        assert "Focus the summary on: auth" in prompt   # and the query focused it
        return "  auth uses jwt rotation  "             # whitespace proves the stage strips it


def _repo_with(*memories: Memory) -> InMemoryMemoryRepository:
    repo = InMemoryMemoryRepository()
    for memory in memories:
        repo.add(memory)
    return repo


def test_generator_fills_a_query_focused_summary_alongside_the_grouping():
    repo = _repo_with(Memory.create("auth jwt rotation", type="decision", project="api"))
    pipeline = build_recall_pipeline(repo, generator=_EchoGenerator())
    bundle = pipeline.run(RecallRequest(project="api", query="auth"))

    assert bundle.summary == "auth uses jwt rotation"  # stripped
    assert bundle.total == 1  # the structured grouping is still present alongside the summary
