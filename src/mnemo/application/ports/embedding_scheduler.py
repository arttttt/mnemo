"""Port: ensure a stored memory gets its embedding — inline or deferred.

Decouples the write use case from *when* the vector is computed. The use case
inserts the memory (pending) and calls `schedule(id)`; a sync scheduler embeds it
now, an async one defers to a background worker (docs/03-architecture.md).
"""
from typing import Protocol


class EmbeddingSchedulerPort(Protocol):
    def schedule(self, memory_id: str) -> None: ...
