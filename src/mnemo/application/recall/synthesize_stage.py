"""Stage: synthesize the grouped bundle into a concise prose digest — the LLM stage.

The one place recall needs text synthesis. It turns the assembled bundle into a focused
prompt and fills ``summary`` (the generator loads and frees itself behind its port). A
no-op on an empty bundle (nothing to summarize).
"""
from __future__ import annotations

from dataclasses import replace

from llmkit.ports.generator import Generator

from mnemo.application.pipeline.context import PipelineContext
from mnemo.application.recall.bundle import RECALL
from mnemo.application.recall.request import RECALL_REQUEST
from mnemo.application.recall.synthesis_prompt import build_synthesis_prompt


class SynthesizeStage:
    key = "synthesize"
    requires = frozenset({RECALL.name})
    provides = frozenset({RECALL.name})  # enriches the bundle in place (fills `summary`)

    def __init__(self, generator: Generator, *, max_tokens: int = 512) -> None:
        self._generator = generator
        self._max_tokens = max_tokens

    def run(self, ctx: PipelineContext) -> PipelineContext:
        bundle = ctx.get(RECALL)
        if bundle.total == 0:
            return ctx  # nothing to summarize
        prompt = build_synthesis_prompt(bundle, query=ctx.get(RECALL_REQUEST).query)
        summary = self._generator.generate(prompt, max_tokens=self._max_tokens)
        return ctx.set(RECALL, replace(bundle, summary=summary.strip()))
