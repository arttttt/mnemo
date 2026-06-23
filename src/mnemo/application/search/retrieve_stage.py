"""Stage: rank memories against the search query — the relevance step, always present.

Reads the seeded ``SEARCH_REQUEST`` and runs the hybrid retrieval — embed the query, then
the dense + lexical (FTS) legs fused, filtered by the request's ``criteria`` and capped at
its ``limit`` — exactly as the old imperative search did. Produces the scored candidates
WITH their scores kept (``RETRIEVED``), for the presentation stage. The first stage of the
retrieve -> (rerank?) -> present pipeline, and the seam where an optional rerank stage will
re-order ``RETRIEVED`` in place before presentation.
"""
from __future__ import annotations

from mnemo.application.pipeline.context import PipelineContext
from mnemo.application.pipeline.slot import Slot
from mnemo.application.ports.embedder import TextEmbedder
from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.retrieval import Retrieval
from mnemo.application.scored_memory import ScoredMemory
from mnemo.application.search.request import SEARCH_REQUEST

RETRIEVED: Slot[tuple[ScoredMemory, ...]] = Slot("retrieved")


class RetrieveStage:
    key = "retrieve"
    requires: frozenset[str] = frozenset()
    provides = frozenset({RETRIEVED.name})

    def __init__(self, repository: MemoryRepository, embedder: TextEmbedder) -> None:
        self._repository = repository
        self._embedder = embedder

    def run(self, ctx: PipelineContext) -> PipelineContext:
        request = ctx.get(SEARCH_REQUEST)
        retrieval = Retrieval(
            criteria=request.criteria,
            limit=request.limit,
            text=request.query,
            vector=self._embedder.encode(request.query),
        )
        scored = self._repository.retrieve(retrieval)
        return ctx.set(RETRIEVED, tuple(scored))
