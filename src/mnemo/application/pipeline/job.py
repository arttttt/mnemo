"""The input to a consolidation run — what goes into the pipeline.

Immutable. It carries only the data inputs (the triggering ``seeds`` and the scope to
consolidate within); per-stage tuning lives on each stage, not here. ``JOB`` is the
slot under which the runner seeds the job, so any stage can read it without declaring
it as a requirement.
"""
from __future__ import annotations

from dataclasses import dataclass

from mnemo.application.pipeline.slot import Slot
from mnemo.domain.memory import Memory
from mnemo.domain.scope import Scope


@dataclass(frozen=True)
class Job:
    seeds: tuple[Memory, ...]
    scope: Scope
    project: str | None = None


JOB: Slot[Job] = Slot("job")
