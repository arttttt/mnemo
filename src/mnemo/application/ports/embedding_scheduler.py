"""Port: ensure a stored memory gets its embedding — inline or deferred.

Decouples the write use case from *when* the vector is computed. The use case
inserts the memory (pending) and calls `schedule(id)`; a sync scheduler embeds it
now, an async one defers to a background worker (docs/03-architecture.md). It also
exposes the embedder's window so the use case can reject oversize content on the
hot path (a cheap token count, not the encode) without depending on the embedder
directly.
"""
from typing import Protocol


class EmbeddingSchedulerPort(Protocol):
    @property
    def max_input(self) -> int: ...

    def count_tokens(self, text: str) -> int: ...

    def schedule(self, memory_id: str) -> None: ...
