"""Embed inline — compute and store the vector immediately.

For the one-shot CLI (the process exits, so there is no worker to defer to) and
offline tests.
"""
from __future__ import annotations

from mnemo.application.ports.embedder import TextEmbedder
from mnemo.application.ports.embedding_queue import EmbeddingQueue


class SyncEmbeddingScheduler:
    def __init__(self, embedder: TextEmbedder, repository: EmbeddingQueue) -> None:
        self._embedder = embedder
        self._repository = repository

    def schedule(self, memory_id: str) -> None:
        content = self._repository.content_for(memory_id)
        if content is None:  # deleted before embedding — nothing to do
            return
        self._repository.set_vector(memory_id, self._embedder.encode(content))
