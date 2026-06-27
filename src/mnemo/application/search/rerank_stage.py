"""Stage: optionally re-order the retrieved pool with a cross-encoder, gated on confidence.

Slots BETWEEN retrieve and present. It only fires when the ``RerankPolicy`` judges the result
ambiguous (a weak top hit or channel disagreement, read from ``SIGNALS``) — a confident,
agreeing result is left exactly as the fuser ranked it, so the expensive cross-encoder pass is
spent only where it has headroom. When it does fire, it scores each candidate's content against
the query and re-orders ``RETRIEVED`` in place; presentation (which trims to the user's limit)
is unchanged. The same optional-refinement shape as recall's ``RerankStage``.
"""
from __future__ import annotations

from llmkit.ports.reranker import Reranker

from mnemo.application.pipeline.context import PipelineContext
from mnemo.application.search.rerank_policy import RerankPolicy
from mnemo.application.search.request import SEARCH_REQUEST
from mnemo.application.search.retrieve_stage import RETRIEVED, SIGNALS


class RerankStage:
    key = "rerank"
    requires = frozenset({RETRIEVED.name, SIGNALS.name, SEARCH_REQUEST.name})
    provides = frozenset({RETRIEVED.name})  # re-orders RETRIEVED in place

    def __init__(self, reranker: Reranker, policy: RerankPolicy) -> None:
        self._reranker = reranker
        self._policy = policy

    def run(self, ctx: PipelineContext) -> PipelineContext:
        pool = ctx.get(RETRIEVED)
        if not pool or not self._policy.should_rerank(ctx.get(SIGNALS)):
            return ctx  # nothing to rank, or a confident result the fuser already ordered well
        query = ctx.get(SEARCH_REQUEST).query
        scores = self._reranker.rank(query, [item.memory.content for item in pool])
        ranked = tuple(
            item
            for item, _ in sorted(zip(pool, scores), key=lambda pair: pair[1], reverse=True)
        )
        return ctx.set(RETRIEVED, ranked)
