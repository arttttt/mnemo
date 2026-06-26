"""Stage: retrieve a project's most query-relevant active memories — the relevance step.

Reads the seeded ``RECALL_REQUEST`` (the intake) and runs the same hybrid retrieval as
``search`` — embed the query, then dense + lexical retrieve — so recall is grounded in the
memories that actually answer the query, not the whole project dumped in. The first stage of
the embedder -> reranker -> generator pipeline. Project-scoped (the named project plus
globally-scoped memories), capped at the request's ``limit``.
"""
from __future__ import annotations

from mnemo.application.fusion.fuser import Fuser
from mnemo.application.pipeline.context import PipelineContext
from mnemo.application.pipeline.slot import Slot
from mnemo.application.ports.embedder import TextEmbedder
from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.recall.request import RECALL_REQUEST
from mnemo.application.retrieval import Retrieval
from mnemo.application.search_criteria import SearchCriteria
from mnemo.domain.memory import Memory

GATHERED: Slot[tuple[Memory, ...]] = Slot("gathered")


class GatherStage:
    key = "gather"
    requires: frozenset[str] = frozenset()
    provides = frozenset({GATHERED.name})

    def __init__(self, repository: MemoryRepository, embedder: TextEmbedder) -> None:
        self._repository = repository
        self._embedder = embedder
        self._fuser = Fuser()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        request = ctx.get(RECALL_REQUEST)
        criteria = SearchCriteria(scope="project", project=request.project)
        retrieval = Retrieval(
            criteria=criteria,
            limit=request.limit,
            text=request.query,
            vector=self._embedder.encode(request.query),
        )
        channels = self._repository.retrieve_channels(retrieval)
        fused = self._fuser.fuse(channels, request.limit)
        return ctx.set(GATHERED, tuple(item.memory for item in fused.pool))
