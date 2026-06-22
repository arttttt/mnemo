"""Recall a project's memory as a query-focused bundle (and a written answer, with a generator).

Builds and runs the recall pipeline once: the embedder retrieves the memories most relevant to
the query (the relevance step), an optional reranker re-orders them, and an optional generator
synthesizes an answer — without a generator recall returns the structured grouping. No generator
runs until synthesis. Exposed both as the ``recall`` MCP tool (the one opt-in LLM read tool) and
the CLI.
"""
from __future__ import annotations

from llmkit.ports.generator import Generator
from llmkit.ports.reranker import Reranker

from mnemo.application.ports.embedder import TextEmbedder
from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.recall.builder import build_recall_pipeline
from mnemo.application.recall.bundle import RecallBundle
from mnemo.application.recall.request import RecallRequest
from mnemo.domain.constants import DEFAULT_RECALL_LIMIT


class RecallProjectUseCaseImpl:
    def __init__(
        self,
        repository: MemoryRepository,
        embedder: TextEmbedder,
        *,
        reranker: Reranker | None = None,
        generator: Generator | None = None,
        rerank_top_k: int = 20,
        generator_max_tokens: int = 512,
    ) -> None:
        self._pipeline = build_recall_pipeline(
            repository,
            embedder,
            reranker=reranker,
            generator=generator,
            top_k=rerank_top_k,
            max_tokens=generator_max_tokens,
        )

    def execute(
        self, *, project: str, query: str, limit: int = DEFAULT_RECALL_LIMIT
    ) -> RecallBundle:
        return self._pipeline.run(RecallRequest(project=project, query=query, limit=limit))
