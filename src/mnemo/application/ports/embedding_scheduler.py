"""Port: ensure a stored memory gets its embedding — inline or deferred.

Decouples the write use case from *when* the vector is computed. The use case
inserts the memory (pending) and calls `schedule(id)`; a sync scheduler embeds it
now, an async one defers to a background worker (docs/03-architecture.md). It also
exposes the embedder's window so the use case can reject oversize content on the
hot path (a cheap token count, not the encode) without depending on the embedder
directly.
"""
from typing import Protocol

from mnemo.application.ports.token_window import TokenWindowPort


class EmbeddingSchedulerPort(TokenWindowPort, Protocol):
    # max_input + count_tokens come from TokenWindowPort (the scheduler forwards them
    # to the embedder); this port adds only the scheduling call.
    def schedule(self, memory_id: str) -> None: ...
