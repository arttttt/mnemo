"""The optional, confidence-gated search reranker, end to end.

Drives the real ``build_search_pipeline`` (real Fuser + a hash-embedder temp SQLite store)
with a FAKE cross-encoder that records its calls, and asserts the gate: an ambiguous result
(a weak top hit) IS reranked and re-ordered, while a confident, channel-agreeing result is
left as the fuser ordered it and the cross-encoder is never invoked.
"""
from __future__ import annotations

import pytest

pytest.importorskip("sqlite_vec")

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.application.fusion.fuser import Fuser
from mnemo.application.search.builder import build_search_pipeline
from mnemo.application.search.rerank_policy import RerankPolicy
from mnemo.application.search.request import SearchRequest
from mnemo.application.search_criteria import SearchCriteria
from mnemo.domain.memory import Memory
from tests.support.sqlite_store import open_store


class _RecordingReranker:
    """A fake cross-encoder: records each call and scores so the LAST document wins, so a
    re-order is observable against the fuser's order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rank(self, query, documents):
        self.calls.append((query, list(documents)))
        # ascending scores → the last document ranks first after the stage re-orders desc
        return [float(i) for i in range(len(documents))]


def _repo_with(tmp_path, embedder, *memories: Memory):
    repo, _ = open_store(tmp_path, embedder.dim, projects=("api",))
    for memory in memories:
        repo.add(memory)
        repo.set_vector(memory.id, embedder.encode(memory.content))
    return repo


def _request(query: str, *, limit: int = 10) -> SearchRequest:
    return SearchRequest(
        criteria=SearchCriteria(scope="project", project="api"), query=query, limit=limit
    )


def test_reranks_and_reorders_an_ambiguous_result(tmp_path):
    embedder = HashEmbedder()
    repo = _repo_with(
        tmp_path,
        embedder,
        Memory.create("redis cache eviction policy", type="decision", project="api"),
        Memory.create("postgres migration plan", type="decision", project="api"),
        Memory.create("jwt refresh rotation note", type="learning", project="api"),
    )
    reranker = _RecordingReranker()
    # floor=1.0 forces the gate open for any real hit (top-1 similarity < 1.0 for a non-exact
    # query) — the ambiguous branch.
    pipeline = build_search_pipeline(
        repo, embedder, Fuser(), reranker=reranker, policy=RerankPolicy(dense_top1_floor=1.0)
    )
    fused = build_search_pipeline(repo, embedder, Fuser()).run(_request("cache and migration"))

    results = pipeline.run(_request("cache and migration"))

    assert reranker.calls  # the cross-encoder WAS invoked
    assert reranker.calls[0][0] == "cache and migration"  # with the query
    # the recording reranker promotes the LAST candidate, so the order changes vs the fuser's
    assert [r.id for r in results] != [r.id for r in fused]
    assert results[0].id == fused[-1].id  # last fuser candidate is now first


def test_confident_result_is_not_reranked(tmp_path):
    embedder = HashEmbedder()
    exact = Memory.create("redis cache eviction policy", type="decision", project="api")
    repo = _repo_with(
        tmp_path,
        embedder,
        exact,
        Memory.create("postgres migration plan", type="decision", project="api"),
    )
    reranker = _RecordingReranker()
    pipeline = build_search_pipeline(
        repo, embedder, Fuser(), reranker=reranker, policy=RerankPolicy(dense_top1_floor=0.45)
    )

    # an EXACT-content query: dense top-1 similarity is 1.0 and both legs rank the same id #1,
    # so the gate stays shut.
    results = pipeline.run(_request("redis cache eviction policy"))

    assert reranker.calls == []  # the cross-encoder was NEVER invoked
    assert results[0].id == exact.id  # the confident hit stays on top, fuser order untouched
