"""Stage: rerank the gathered memories by relevance to the recall query, keep the top-K.

Only meaningful with a ``query`` to rank against. It scores each gathered memory's
content against the query (the reranker loads and frees itself behind its port) and
overwrites ``GATHERED`` with the most-relevant few — so the downstream ``AssembleStage``
is unchanged. With no query it is a no-op (leaves the gather as-is).
"""
from __future__ import annotations

from llmkit.ports.reranker import Reranker

from mnemo.application.pipeline.context import PipelineContext
from mnemo.application.recall.gather_stage import GATHERED
from mnemo.application.recall.request import RECALL_REQUEST


class RerankStage:
    key = "rerank"
    requires = frozenset({GATHERED.name})
    provides = frozenset({GATHERED.name})  # transforms GATHERED in place (re-ranked, trimmed)

    def __init__(self, reranker: Reranker, *, top_k: int) -> None:
        self._reranker = reranker
        self._top_k = top_k

    def run(self, ctx: PipelineContext) -> PipelineContext:
        request = ctx.get(RECALL_REQUEST)
        gathered = ctx.get(GATHERED)
        if not gathered:
            return ctx  # nothing to rank
        documents = [memory.content for memory in gathered]
        scores = self._reranker.rank(request.query, documents)
        ranked = [
            memory
            for memory, _ in sorted(zip(gathered, scores), key=lambda pair: pair[1], reverse=True)
        ]
        return ctx.set(GATHERED, tuple(ranked[: self._top_k]))
