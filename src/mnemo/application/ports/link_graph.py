"""Port: the typed-link graph between memories (provenance edges).

The supersedes edge is written automatically inside the store on a topic_key
upsert; this explicit read/write API is the seam for the future neighbors/get
tools. One store implementation realizes this alongside MemoryRepository and
EmbeddingQueue.
"""
from __future__ import annotations

from typing import Protocol

from mnemo.domain.link import Link


class LinkGraph(Protocol):
    def add_link(self, link: Link) -> None: ...

    def links_for(self, memory_id: str) -> list[Link]: ...
