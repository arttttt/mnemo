"""The pipeline runner — a trivial sequential composition of stages.

Generic over its output slot, so the runner itself knows nothing about consolidation
artifacts. At construction it validates the chain (each stage's ``requires`` must
already be available from an earlier stage or the seeded job) and that the output slot
is produced; per run it checks each stage actually filled what it promised. A stage
error propagates — aborting the whole job and applying nothing — by design: the caller
never applies a plan on failure.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, TypeVar

from mnemo.application.pipeline.context import PipelineContext
from mnemo.application.pipeline.errors import PipelineError
from mnemo.application.pipeline.job import JOB, Job
from mnemo.application.pipeline.slot import Slot
from mnemo.application.pipeline.stage import PipelineStage

T = TypeVar("T")


class Pipeline(Generic[T]):
    def __init__(self, stages: Sequence[PipelineStage], *, produces: Slot[T]) -> None:
        available = {JOB.name}
        for stage in stages:
            missing = stage.requires - available
            if missing:
                raise PipelineError(f"{stage.key}: missing inputs {sorted(missing)}")
            available |= stage.provides
        if produces.name not in available:
            raise PipelineError(f"pipeline produces no '{produces.name}'")
        self._stages = tuple(stages)
        self._produces = produces

    def run(self, job: Job) -> T:
        ctx = PipelineContext.empty().set(JOB, job)
        for stage in self._stages:
            ctx = stage.run(ctx)
            unmet = stage.provides - ctx.filled
            if unmet:
                raise PipelineError(f"{stage.key}: did not fill {sorted(unmet)}")
        return ctx.get(self._produces)
