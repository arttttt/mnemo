"""Migrate the store to the current embedder.

Rebuilds the store at the embedder's dimension (a no-op if unchanged) and re-embeds
every memory through the scheduler. Content, metadata and links are preserved — only
the vectors are re-derived. Reuses the pending-vector path: set_dimension drops all
vectors to pending, then each is scheduled for re-embedding.
"""
from __future__ import annotations

from mnemo.application.ports.embedder import EmbedderPort
from mnemo.application.ports.embedding_scheduler import EmbeddingSchedulerPort
from mnemo.application.ports.memory_repository import MemoryRepositoryPort


class ReindexMemories:
    def __init__(
        self,
        repository: MemoryRepositoryPort,
        embedder: EmbedderPort,
        scheduler: EmbeddingSchedulerPort,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._scheduler = scheduler

    def execute(self) -> int:
        """Re-embed every memory at the current embedder's dimension; returns the count."""
        self._repository.set_dimension(self._embedder.dim)
        pending = self._repository.next_unembedded(self._repository.pending_count())
        for memory_id in pending:
            self._scheduler.schedule(memory_id)
        return len(pending)
