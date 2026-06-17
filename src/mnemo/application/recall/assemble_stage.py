"""Stage: group gathered memories by type into sections — pure, no model.

The seam for a future LLM synthesis stage: today it structures the bundle
deterministically; later a generation stage can compress these sections into prose.
"""
from __future__ import annotations

from mnemo.application.pipeline.context import PipelineContext
from mnemo.application.recall.bundle import RECALL, RecallBundle, RecallSection
from mnemo.application.recall.gather_stage import GATHERED
from mnemo.application.recall.request import RECALL_REQUEST
from mnemo.domain.memory import Memory


class AssembleStage:
    key = "assemble"
    requires = frozenset({GATHERED.name})
    provides = frozenset({RECALL.name})

    def run(self, ctx: PipelineContext) -> PipelineContext:
        request = ctx.get(RECALL_REQUEST)
        grouped: dict[str, list[Memory]] = {}
        for memory in ctx.get(GATHERED):
            grouped.setdefault(memory.type.value, []).append(memory)
        sections = tuple(
            RecallSection(type=type_value, memories=tuple(memories))
            for type_value, memories in grouped.items()
        )
        return ctx.set(RECALL, RecallBundle(project=request.project, sections=sections))
