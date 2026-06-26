"""Assemble the search pipeline: retrieve -> (rerank?) -> present.

The retrieve stage ranks the query-relevant memories with the hybrid (dense + lexical)
retrieval — the relevance step, always present. The present stage maps them to the public
``SearchResult`` page. A reranker, when wired, will slot BETWEEN the two — re-ordering the
retrieved candidates before presentation — the same optional-refinement shape as recall's
``build_recall_pipeline``. Search has no reranker yet, so the chain is the two ends.
"""
from __future__ import annotations

from mnemo.application.fusion.fuser import Fuser
from mnemo.application.pipeline.pipeline import Pipeline
from mnemo.application.ports.embedder import TextEmbedder
from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.results.search_result import SearchResult
from mnemo.application.search.present_stage import SEARCH_RESULTS, PresentStage
from mnemo.application.search.request import SEARCH_REQUEST, SearchRequest
from mnemo.application.search.retrieve_stage import RetrieveStage


def build_search_pipeline(
    repository: MemoryRepository,
    embedder: TextEmbedder,
    fuser: Fuser,
) -> Pipeline[SearchRequest, list[SearchResult]]:
    stages = [RetrieveStage(repository, embedder, fuser), PresentStage()]
    return Pipeline(stages, intake=SEARCH_REQUEST, produces=SEARCH_RESULTS)
