"""Assemble the search pipeline: retrieve -> (rerank?) -> present.

The retrieve stage ranks the query-relevant memories with the hybrid (dense + lexical)
retrieval — the relevance step, always present. The present stage maps them to the public
``SearchResult`` page. With a reranker wired, a confidence-gated ``RerankStage`` slots BETWEEN
the two — re-ordering an over-fetched candidate pool before presentation trims it — the same
optional-refinement shape as recall's ``build_recall_pipeline``. Without one the chain is the
two ends and retrieval fuses straight to the page size.
"""
from __future__ import annotations

from llmkit.ports.reranker import Reranker

from mnemo.application.fusion.fuser import Fuser
from mnemo.application.pipeline.pipeline import Pipeline
from mnemo.application.ports.embedder import TextEmbedder
from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.results.search_result import SearchResult
from mnemo.application.search.present_stage import SEARCH_RESULTS, PresentStage
from mnemo.application.search.rerank_policy import RerankPolicy
from mnemo.application.search.rerank_stage import RerankStage
from mnemo.application.search.request import SEARCH_REQUEST, SearchRequest
from mnemo.application.search.retrieve_stage import RetrieveStage

def build_search_pipeline(
    repository: MemoryRepository,
    embedder: TextEmbedder,
    fuser: Fuser,
    *,
    reranker: Reranker | None = None,
    policy: RerankPolicy | None = None,
) -> Pipeline[SearchRequest, list[SearchResult]]:
    if reranker is not None:
        stages = [
            RetrieveStage(repository, embedder, fuser, over_fetch=True),
            RerankStage(reranker, policy or RerankPolicy()),
            PresentStage(),
        ]
    else:
        stages = [RetrieveStage(repository, embedder, fuser), PresentStage()]
    return Pipeline(stages, intake=SEARCH_REQUEST, produces=SEARCH_RESULTS)
