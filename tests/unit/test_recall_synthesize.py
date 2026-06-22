"""Synthesized recall — a generator turns the grouped bundle into a query-focused summary."""
from __future__ import annotations

import pytest

pytest.importorskip("sqlite_vec")

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.application.recall.builder import build_recall_pipeline
from mnemo.application.recall.request import RecallRequest
from mnemo.domain.memory import Memory
from tests.support.sqlite_store import open_store


class _EchoGenerator:
    """A deterministic stand-in: asserts the prompt was built, returns a fixed summary."""

    def generate(self, prompt, *, max_tokens):
        assert "auth jwt rotation" in prompt   # the gathered memory reached the prompt
        assert "Question: auth" in prompt      # and the query focused it
        return "  auth uses jwt rotation  "    # whitespace proves the stage strips it


def _repo_with(tmp_path, embedder, *memories: Memory):
    repo, _ = open_store(tmp_path, embedder.dim, projects=("api",))
    for memory in memories:
        repo.add(memory)
        repo.set_vector(memory.id, embedder.encode(memory.content))
    return repo


def test_generator_fills_a_query_focused_summary_alongside_the_grouping(tmp_path):
    embedder = HashEmbedder()
    repo = _repo_with(tmp_path, embedder, Memory.create("auth jwt rotation", type="decision", project="api"))
    pipeline = build_recall_pipeline(repo, embedder, generator=_EchoGenerator())
    bundle = pipeline.run(RecallRequest(project="api", query="auth"))

    assert bundle.summary == "auth uses jwt rotation"  # stripped
    assert bundle.total == 1  # the structured grouping is still present alongside the summary


class _RecordingGenerator:
    """Records the max_tokens the synthesize stage asks for."""

    def __init__(self) -> None:
        self.max_tokens: int | None = None

    def generate(self, prompt, *, max_tokens):
        self.max_tokens = max_tokens
        return "summary"


def test_max_tokens_threads_through_to_the_generator(tmp_path):
    embedder = HashEmbedder()
    repo = _repo_with(tmp_path, embedder, Memory.create("auth jwt rotation", type="decision", project="api"))
    generator = _RecordingGenerator()
    pipeline = build_recall_pipeline(repo, embedder, generator=generator, max_tokens=123)
    pipeline.run(RecallRequest(project="api", query="auth"))

    assert generator.max_tokens == 123  # not the hardcoded 512
