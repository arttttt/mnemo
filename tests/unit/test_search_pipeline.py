"""The search pipeline (retrieve -> present): the hybrid relevance step mapped to the public
``SearchResult`` page. Memories must be embedded for the dense leg to find them; the present
stage shapes the scored candidates into result DTOs, preserving order.
"""
from __future__ import annotations

import pytest

pytest.importorskip("sqlite_vec")

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.application.fusion.fuser import Fuser
from mnemo.application.results.search_result import SearchResult
from mnemo.application.search.builder import build_search_pipeline
from mnemo.application.search.request import SearchRequest
from mnemo.application.search_criteria import SearchCriteria
from mnemo.domain.memory import Memory
from tests.support.sqlite_store import open_store


def _repo_with(tmp_path, embedder, *memories: Memory):
    repo, _ = open_store(tmp_path, embedder.dim, projects=("api", "other"))
    for memory in memories:
        repo.add(memory)
        repo.set_vector(memory.id, embedder.encode(memory.content))
    return repo


def _request(query: str, *, project: str = "api", limit: int = 10) -> SearchRequest:
    return SearchRequest(
        criteria=SearchCriteria(scope="project", project=project), query=query, limit=limit
    )


def test_returns_search_results_for_the_query(tmp_path):
    embedder = HashEmbedder()
    repo = _repo_with(
        tmp_path,
        embedder,
        Memory.create("jwt refresh rotation", type="decision", project="api"),
        Memory.create("fixed the race", type="learning", project="api"),
    )
    results = build_search_pipeline(repo, embedder, Fuser()).run(_request("jwt refresh rotation"))

    assert results
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(not hasattr(r, "score") for r in results)  # search hits carry no relevance score
    assert "jwt refresh rotation" in {r.content for r in results}


def test_scopes_to_the_project_but_includes_global_memories(tmp_path):
    embedder = HashEmbedder()
    repo = _repo_with(
        tmp_path,
        embedder,
        Memory.create("api decision", type="decision", project="api"),
        Memory.create("other project", type="decision", project="other"),
        Memory.create("a global rule", type="rule", scope="global"),
    )
    results = build_search_pipeline(repo, embedder, Fuser()).run(_request("anything"))

    contents = {r.content for r in results}
    assert contents == {"api decision", "a global rule"}  # 'other' excluded, global kept


def test_limit_caps_the_number_of_results(tmp_path):
    embedder = HashEmbedder()
    repo = _repo_with(
        tmp_path,
        embedder,
        *[Memory.create(f"note {i}", type="working-notes", project="api") for i in range(5)],
    )
    results = build_search_pipeline(repo, embedder, Fuser()).run(_request("notes", limit=2))

    assert len(results) == 2
