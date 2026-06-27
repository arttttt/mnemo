"""Focus recall — reranks the gathered memories by relevance to the query, keeps top-K."""
from __future__ import annotations

import pytest

pytest.importorskip("sqlite_vec")

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.application.fusion.fuser import Fuser
from mnemo.application.recall.builder import build_recall_pipeline
from mnemo.application.recall.request import RecallRequest
from mnemo.domain.memory import Memory
from tests.support.sqlite_store import open_store


class _KeywordReranker:
    """A deterministic stand-in for a cross-encoder: scores by query-word overlap."""

    def rank(self, query, documents):
        words = set(query.lower().split())
        return [float(len(words & set(doc.lower().split()))) for doc in documents]


def _repo_with(tmp_path, embedder, *memories: Memory):
    repo, _ = open_store(tmp_path, embedder.dim, projects=("api",))
    for memory in memories:
        repo.add(memory)
        repo.set_vector(memory.id, embedder.encode(memory.content))
    return repo


def test_query_reranks_gathered_by_relevance_and_trims_to_top_k(tmp_path):
    embedder = HashEmbedder()
    repo = _repo_with(
        tmp_path,
        embedder,
        Memory.create("auth jwt rotation decision", type="decision", project="api"),
        Memory.create("logging format change", type="decision", project="api"),
        Memory.create("auth session cookie note", type="working-notes", project="api"),
    )
    pipeline = build_recall_pipeline(repo, embedder, Fuser(), reranker=_KeywordReranker(), top_k=2)
    bundle = pipeline.run(RecallRequest(project="api", query="auth"))

    # the two 'auth' memories outrank the logging one, which is trimmed by top_k
    contents = {m.content for section in bundle.sections for m in section.memories}
    assert bundle.total == 2
    assert contents == {"auth jwt rotation decision", "auth session cookie note"}
