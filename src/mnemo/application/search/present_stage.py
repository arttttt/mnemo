"""Stage: turn the scored candidates into the public ``SearchResult`` page.

The terminal stage of the search pipeline: maps each retrieved ``ScoredMemory`` to a
``SearchResult`` DTO (rounding the score), preserving order. Kept separate from retrieval
so an optional rerank stage can re-order ``RETRIEVED`` in place beforehand — presentation
is then unchanged, the same degradation shape as recall's assemble-after-rerank.
"""
from __future__ import annotations

from mnemo.application.pipeline.context import PipelineContext
from mnemo.application.pipeline.slot import Slot
from mnemo.application.results.search_result import SearchResult
from mnemo.application.search.retrieve_stage import RETRIEVED

SEARCH_RESULTS: Slot[list[SearchResult]] = Slot("search_results")


class PresentStage:
    key = "present"
    requires = frozenset({RETRIEVED.name})
    provides = frozenset({SEARCH_RESULTS.name})

    def run(self, ctx: PipelineContext) -> PipelineContext:
        scored = ctx.get(RETRIEVED)
        results = [
            SearchResult(
                id=item.memory.id,
                score=round(item.score, 4),
                type=item.memory.type.value,
                scope=item.memory.scope.value,
                project=item.memory.project,
                content=item.memory.content,
                related_files=item.memory.related_files,
                created_at=item.memory.created_at,
            )
            for item in scored
        ]
        return ctx.set(SEARCH_RESULTS, results)
