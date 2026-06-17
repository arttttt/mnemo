"""Recall a project's memory as a structured bundle (model-free).

Builds and runs the recall pipeline (gather → group); no LLM and no embedding — it
reads the store's browse path. Kept off the agent-facing MCP surface for now: a useful
(non-dumping) recall needs LLM synthesis (see docs/02-requirements.md FR-11), so this
structured version is a CLI dev/debug affordance until a generation stage lands.
"""
from __future__ import annotations

from mnemo.application.ports.memory_repository import MemoryRepositoryPort
from mnemo.application.recall.builder import build_recall_pipeline
from mnemo.application.recall.bundle import RecallBundle
from mnemo.application.recall.request import RecallRequest


class RecallProject:
    def __init__(self, repository: MemoryRepositoryPort) -> None:
        self._pipeline = build_recall_pipeline(repository)

    def execute(self, *, project: str, limit: int = 50) -> RecallBundle:
        return self._pipeline.run(RecallRequest(project=project, limit=limit))
