"""Migrate the store to the current embedder.

Rebuilds the store at the embedder's dimension (a no-op if unchanged) and re-embeds
every memory through the scheduler. Content, metadata and links are preserved — only
the vectors are re-derived. Reuses the pending-vector path: set_dimension drops all
vectors to pending, then each is scheduled for re-embedding.
"""
from __future__ import annotations

from mnemo.application.ports.embedder import EmbedderPort
from mnemo.application.ports.embedding_queue import EmbeddingQueuePort
from mnemo.application.ports.embedding_scheduler import EmbeddingSchedulerPort

_REINDEX_PAGE = 256  # rows fetched per scan while draining the pending set


class ReindexMemories:
    def __init__(
        self,
        repository: EmbeddingQueuePort,
        embedder: EmbedderPort,
        scheduler: EmbeddingSchedulerPort,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._scheduler = scheduler

    def execute(self) -> int:
        """Re-embed every memory at the current embedder's dimension; returns the count."""
        self._repository.set_dimension(self._embedder.dim)
        # Page through the pending rows until none remain, rather than trusting one
        # pre-read count. reindex runs with the inline (sync) scheduler — the service is
        # stopped first — so each scheduled id is embedded at once and leaves the pending
        # set, which is what lets the scan advance and the loop terminate.
        reindexed = 0
        while True:
            batch = self._repository.next_unembedded(_REINDEX_PAGE)
            if not batch:
                return reindexed
            for memory_id in batch:
                self._scheduler.schedule(memory_id)
                reindexed += 1
