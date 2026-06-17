"""The pipeline runner — a trivial sequential composition of stages.

Generic over its input and output slots, so the runner itself knows nothing about any
particular pipeline. At construction it validates the chain (each stage's ``requires``
must already be available from an earlier stage or the seeded ``intake``) and that the
output slot is produced; per run it checks each stage actually filled what it promised.
A stage error propagates — aborting the whole run and applying nothing — by design: the
caller never applies a result on failure.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, TypeVar

from mnemo.application.pipeline.context import PipelineContext
from mnemo.application.pipeline.errors import PipelineError
from mnemo.application.pipeline.slot import Slot
from mnemo.application.pipeline.stage import PipelineStage

R = TypeVar("R")
T = TypeVar("T")


class Pipeline(Generic[R, T]):
    def __init__(
        self,
        stages: Sequence[PipelineStage],
        *,
        intake: Slot[R],
        produces: Slot[T],
    ) -> None:
        available = {intake.name}
        for stage in stages:
            missing = stage.requires - available
            if missing:
                raise PipelineError(f"{stage.key}: missing inputs {sorted(missing)}")
            available |= stage.provides
        if produces.name not in available:
            raise PipelineError(f"pipeline produces no '{produces.name}'")
        self._stages = tuple(stages)
        self._intake = intake
        self._produces = produces

    def run(self, request: R) -> T:
        ctx = PipelineContext.empty().set(self._intake, request)
        for stage in self._stages:
            ctx = stage.run(ctx)
            unmet = stage.provides - ctx.filled
            if unmet:
                raise PipelineError(f"{stage.key}: did not fill {sorted(unmet)}")
        return ctx.get(self._produces)
