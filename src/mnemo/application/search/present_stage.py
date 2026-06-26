"""Stage: turn the scored candidates into the public ``SearchResult`` page.

The terminal stage of the search pipeline: maps each retrieved ``ScoredMemory`` to a
``SearchResult`` DTO, preserving order, after trimming to the user's ``limit`` — retrieval
(and an optional reranker) work over a wider candidate pool, so the page is cut to size HERE.
Kept separate from retrieval so an optional rerank stage can re-order ``RETRIEVED`` in place
beforehand — presentation is then unchanged, the same degradation shape as recall's
assemble-after-rerank.
"""
from __future__ import annotations

from mnemo.application.pipeline.context import PipelineContext
from mnemo.application.pipeline.slot import Slot
from mnemo.application.results.search_result import SearchResult
from mnemo.application.search.request import SEARCH_REQUEST
from mnemo.application.search.retrieve_stage import RETRIEVED

SEARCH_RESULTS: Slot[list[SearchResult]] = Slot("search_results")


class PresentStage:
    key = "present"
    requires = frozenset({RETRIEVED.name, SEARCH_REQUEST.name})
    provides = frozenset({SEARCH_RESULTS.name})

    def run(self, ctx: PipelineContext) -> PipelineContext:
        scored = ctx.get(RETRIEVED)[: ctx.get(SEARCH_REQUEST).limit]
        results = [
            SearchResult(
                id=item.memory.id,
                type=item.memory.type.value,
                scope=item.memory.scope.value,
                project=item.memory.project,
                content=item.memory.content,
                related_files=item.memory.related_files,
                created_at=item.memory.created_at,
                topic_key=item.memory.topic_key,
            )
            for item in scored
        ]
        return ctx.set(SEARCH_RESULTS, results)
