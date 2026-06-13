"""Copy memories from one store to another, re-embedding so the target is
consistent with its own embedder. Idempotent: records already present in the
target (matched by id) are skipped, so re-running is safe and never duplicates.
"""
from __future__ import annotations

from mnemo.application.ports.embedder import EmbedderPort
from mnemo.application.ports.memory_repository import MemoryRepositoryPort
from mnemo.application.results.migration_result import MigrationResult


class MigrateMemories:
    def __init__(
        self,
        source: MemoryRepositoryPort,
        target: MemoryRepositoryPort,
        embedder: EmbedderPort,
    ) -> None:
        self._source = source
        self._target = target
        self._embedder = embedder

    def execute(self) -> MigrationResult:
        existing = {memory.id for memory in self._target.list_all()}
        source = self._source.list_all()
        added = 0
        for memory in source:
            if memory.id in existing:
                continue
            self._target.add(memory, self._embedder.encode(memory.content))
            added += 1
        return MigrationResult(
            source_total=len(source),
            added=added,
            skipped=len(source) - added,
        )
