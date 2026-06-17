"""The pipeline's output: an ordered, idempotent plan of store operations.

Produced by the terminal stage and applied later by a separate executor — the pipeline
itself never writes to the store. ``PLAN`` is the conventional output slot a
consolidation pipeline declares.
"""
from __future__ import annotations

from dataclasses import dataclass

from mnemo.application.pipeline.operations import Operation
from mnemo.application.pipeline.slot import Slot


@dataclass(frozen=True)
class Plan:
    operations: tuple[Operation, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.operations


PLAN: Slot["Plan"] = Slot("plan")
