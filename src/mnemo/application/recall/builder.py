"""Assemble the recall pipeline — model-free: gather a project's memory, then group it.

A new consolidation task is sketched the same way: one builder listing its blocks.
When a synthesis model is chosen, a generation stage slots in after ``AssembleStage``.
"""
from __future__ import annotations

from mnemo.application.pipeline.pipeline import Pipeline
from mnemo.application.ports.memory_repository import MemoryRepositoryPort
from mnemo.application.recall.assemble_stage import AssembleStage
from mnemo.application.recall.bundle import RECALL, RecallBundle
from mnemo.application.recall.gather_stage import GatherStage
from mnemo.application.recall.request import RECALL_REQUEST, RecallRequest


def build_recall_pipeline(
    repository: MemoryRepositoryPort,
) -> Pipeline[RecallRequest, RecallBundle]:
    return Pipeline(
        [GatherStage(repository), AssembleStage()],
        intake=RECALL_REQUEST,
        produces=RECALL,
    )
