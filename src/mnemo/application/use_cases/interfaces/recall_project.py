"""Interface for the recall use case."""
from __future__ import annotations

from typing import Protocol

from mnemo.application.recall.bundle import RecallBundle


class RecallProjectUseCase(Protocol):
    def execute(self, *, project: str, limit: int = 50) -> RecallBundle: ...
