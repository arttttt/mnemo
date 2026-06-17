"""Recall a project's memory as a query-focused bundle (and a summary, with a generator).

Builds and runs the recall pipeline once; the reranker orders the gathered memories by
the query and the generator synthesizes a summary, both optional — without them recall
returns the structured grouping. No model runs until a stage opens its session. Kept off
the agent-facing MCP surface for now: a useful (non-dumping) recall needs LLM synthesis
(see docs/02-requirements.md FR-11), so this is a CLI dev/debug affordance until the
generator is finalized.
"""
from __future__ import annotations

from mnemo.application.ports.generator import GeneratorPort
from mnemo.application.ports.memory_repository import MemoryRepositoryPort
from mnemo.application.ports.reranker import RerankerPort
from mnemo.application.recall.builder import build_recall_pipeline
from mnemo.application.recall.bundle import RecallBundle
from mnemo.application.recall.request import RecallRequest


class RecallProject:
    def __init__(
        self,
        repository: MemoryRepositoryPort,
        *,
        reranker: RerankerPort | None = None,
        generator: GeneratorPort | None = None,
        rerank_top_k: int = 20,
    ) -> None:
        self._pipeline = build_recall_pipeline(
            repository, reranker=reranker, generator=generator, top_k=rerank_top_k
        )

    def execute(self, *, project: str, query: str, limit: int = 50) -> RecallBundle:
        return self._pipeline.run(RecallRequest(project=project, query=query, limit=limit))
