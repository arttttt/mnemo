"""Assemble the recall pipeline: embedder -> reranker -> generator, the refinements optional.

The gather stage retrieves the query-relevant memories with the embedder (the relevance step,
always present). With a reranker it inserts ``RerankStage`` to re-order that gather by the
query; with a generator it appends ``SynthesizeStage`` for a written answer. Without a
generator recall is the structured grouping — the same degradation philosophy as consolidation
(each refinement earns its place when its model is configured).
"""
from __future__ import annotations

from llmkit.ports.generator import Generator
from llmkit.ports.reranker import Reranker

from mnemo.application.fusion.fuser import Fuser
from mnemo.application.pipeline.pipeline import Pipeline
from mnemo.application.ports.embedder import TextEmbedder
from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.recall.assemble_stage import AssembleStage
from mnemo.application.recall.bundle import RECALL, RecallBundle
from mnemo.application.recall.gather_stage import GatherStage
from mnemo.application.recall.rerank_stage import RerankStage
from mnemo.application.recall.request import RECALL_REQUEST, RecallRequest
from mnemo.application.recall.synthesize_stage import SynthesizeStage


def build_recall_pipeline(
    repository: MemoryRepository,
    embedder: TextEmbedder,
    fuser: Fuser,
    *,
    reranker: Reranker | None = None,
    generator: Generator | None = None,
    top_k: int = 20,
    max_tokens: int = 512,
) -> Pipeline[RecallRequest, RecallBundle]:
    stages = [GatherStage(repository, embedder, fuser)]
    if reranker is not None:
        stages.append(RerankStage(reranker, top_k=top_k))
    stages.append(AssembleStage())
    if generator is not None:
        stages.append(SynthesizeStage(generator, max_tokens=max_tokens))
    return Pipeline(stages, intake=RECALL_REQUEST, produces=RECALL)
