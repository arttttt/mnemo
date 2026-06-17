"""Port: one composable step of a pipeline (a "block").

A stage has a single responsibility: read the slots it ``requires``, do its work
(using at most one model behind its own port), and return a context extended with the
slots it ``provides``. Loading/unloading a heavy model is the stage's own concern (it
scopes the model with a ``with``), so the port stays minimal — a lightweight stage is
never forced to implement lifecycle hooks it does not need.
"""
from __future__ import annotations

from typing import Protocol

from mnemo.application.pipeline.context import PipelineContext


class PipelineStage(Protocol):
    key: str
    requires: frozenset[str]
    provides: frozenset[str]

    def run(self, ctx: PipelineContext) -> PipelineContext: ...
