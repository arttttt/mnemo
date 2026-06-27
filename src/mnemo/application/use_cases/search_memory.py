"""Retrieve active memories by similarity within a (soft) scope and filters.

A thin use case over the search pipeline (retrieve -> present): it owns the two things the
pipeline should not — turning the request params into a ``SearchCriteria`` and the
authorization check (fail fast on an unknown / missing project, before any embedding) —
then runs the pipeline. The hybrid ranking and result shaping live in the stages.
"""
from __future__ import annotations

from llmkit.ports.reranker import Reranker

from mnemo.application.fusion.fuser import Fuser
from mnemo.application.ports.embedder import TextEmbedder
from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.project_gate import ProjectGate
from mnemo.application.results.search_result import SearchResult
from mnemo.application.search.builder import build_search_pipeline
from mnemo.application.search.rerank_policy import RerankPolicy
from mnemo.application.search.request import SearchRequest
from mnemo.application.search_criteria import SearchCriteria
from mnemo.domain.memory_type import MemoryType


class SearchMemoryUseCaseImpl:
    def __init__(
        self,
        repository: MemoryRepository,
        embedder: TextEmbedder,
        gate: ProjectGate,
        fuser: Fuser,
        reranker: Reranker | None = None,
        policy: RerankPolicy | None = None,
    ) -> None:
        self._gate = gate
        self._pipeline = build_search_pipeline(
            repository, embedder, fuser, reranker=reranker, policy=policy
        )

    def execute(
        self,
        *,
        query: str,
        scope: str = "project",
        project: str | None = None,
        type: str | None = None,
        tags: list[str] | None = None,
        related_files: list[str] | None = None,
        created_after: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        criteria = SearchCriteria(
            scope=scope,
            project=project,
            type=MemoryType(type) if type else None,
            tags=tuple(tags or ()),
            related_files=tuple(related_files or ()),
            created_after=created_after,
        )
        self._gate.check(criteria.scope, criteria.project)
        return self._pipeline.run(
            SearchRequest(criteria=criteria, query=query, limit=limit)
        )
