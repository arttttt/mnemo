"""Stage: gather a project's active memories, newest first — repository only, no model.

Reads the seeded ``RECALL_REQUEST`` (the intake) and takes the store's filter-only
browse path (a Retrieval with neither text nor vector), so it needs no embedding.
"""
from __future__ import annotations

from mnemo.application.pipeline.context import PipelineContext
from mnemo.application.pipeline.slot import Slot
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

    def __init__(self, repository: MemoryRepository) -> None:
        self._repository = repository

    def run(self, ctx: PipelineContext) -> PipelineContext:
        request = ctx.get(RECALL_REQUEST)
        criteria = SearchCriteria(scope="project", project=request.project)
        scored = self._repository.retrieve(Retrieval(criteria=criteria, limit=request.limit))
        return ctx.set(GATHERED, tuple(item.memory for item in scored))
