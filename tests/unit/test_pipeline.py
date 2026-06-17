"""Pipeline runner — chain validation at construction, postconditions at run."""
from __future__ import annotations

import pytest

from mnemo.application.pipeline.context import PipelineContext
from mnemo.application.pipeline.errors import PipelineError
from mnemo.application.pipeline.job import JOB, Job
from mnemo.application.pipeline.pipeline import Pipeline
from mnemo.application.pipeline.slot import Slot
from mnemo.domain.scope import Scope

NUMBERS: Slot[tuple[int, ...]] = Slot("numbers")
TOTAL: Slot[int] = Slot("total")


class _Produce:
    key = "produce"
    requires: frozenset[str] = frozenset()
    provides = frozenset({NUMBERS.name})

    def __init__(self, values: tuple[int, ...]) -> None:
        self._values = values

    def run(self, ctx: PipelineContext) -> PipelineContext:
        return ctx.set(NUMBERS, self._values)


class _Sum:
    key = "sum"
    requires = frozenset({NUMBERS.name})
    provides = frozenset({TOTAL.name})

    def run(self, ctx: PipelineContext) -> PipelineContext:
        return ctx.set(TOTAL, sum(ctx.get(NUMBERS)))


class _Liar:
    # declares it produces TOTAL but never sets it
    key = "liar"
    requires: frozenset[str] = frozenset()
    provides = frozenset({TOTAL.name})

    def run(self, ctx: PipelineContext) -> PipelineContext:
        return ctx


def _job() -> Job:
    return Job(seeds=(), scope=Scope.PROJECT, project="api")


def test_runs_each_stage_in_order_and_returns_the_output_slot():
    pipe = Pipeline([_Produce((1, 2, 3)), _Sum()], intake=JOB, produces=TOTAL)
    assert pipe.run(_job()) == 6


def test_the_job_is_readable_without_being_declared_a_requirement():
    seen: dict[str, object] = {}

    class _ReadJob:
        key = "read-job"
        requires: frozenset[str] = frozenset()
        provides = frozenset({NUMBERS.name})

        def run(self, ctx: PipelineContext) -> PipelineContext:
            seen["project"] = ctx.get(JOB).project
            return ctx.set(NUMBERS, ())

    Pipeline([_ReadJob(), _Sum()], intake=JOB, produces=TOTAL).run(_job())
    assert seen["project"] == "api"


def test_a_stage_whose_input_no_earlier_stage_produces_is_rejected_on_build():
    with pytest.raises(PipelineError) as exc:
        Pipeline([_Sum(), _Produce((1,))], intake=JOB, produces=TOTAL)  # Sum before Produce
    assert "numbers" in str(exc.value)


def test_a_pipeline_that_never_produces_its_output_is_rejected_on_build():
    with pytest.raises(PipelineError) as exc:
        Pipeline([_Produce((1,))], intake=JOB, produces=TOTAL)  # nobody fills TOTAL
    assert "total" in str(exc.value)


def test_a_stage_that_does_not_fill_what_it_promised_fails_at_run():
    pipe = Pipeline([_Liar()], intake=JOB, produces=TOTAL)
    with pytest.raises(PipelineError) as exc:
        pipe.run(_job())
    assert "total" in str(exc.value)
