"""A typed, directed edge between two memories, with provenance.

Links are created *for* the agent, deterministically, as a by-product of normal
actions (a topic_key upsert writes a `supersedes` edge) — the coding agent never
authors them, and nothing here is semantically inferred (that boundary is a
project axiom; see docs/adr/0001-storage-engine.md, "Not a knowledge graph").
`provenance` records *how* the edge was created, so a caller can tell a certain,
deterministic link from a later, model-inferred guess.
"""
from __future__ import annotations

from dataclasses import dataclass

from mnemo.domain.generators import now
from mnemo.domain.link_type import LinkType


@dataclass(frozen=True)
class Link:
    source_id: str
    target_id: str
    type: LinkType
    provenance: str
    created_at: str

    @classmethod
    def supersedes(cls, *, source_id: str, target_id: str, provenance: str) -> "Link":
        """The successor (`source_id`) supersedes the prior record (`target_id`)."""
        return cls(
            source_id=source_id,
            target_id=target_id,
            type=LinkType.SUPERSEDES,
            provenance=provenance,
            created_at=now(),
        )
