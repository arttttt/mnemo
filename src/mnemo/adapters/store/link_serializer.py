"""(De)serialize a Link to/from a plain dict — shared by the store backends."""
from __future__ import annotations

from mnemo.domain.link import Link
from mnemo.domain.link_type import LinkType


def link_to_dict(link: Link) -> dict:
    return {
        "source_id": link.source_id,
        "target_id": link.target_id,
        "type": link.type.value,
        "provenance": link.provenance,
        "created_at": link.created_at,
    }


def link_from_dict(data: dict) -> Link:
    return Link(
        source_id=data["source_id"],
        target_id=data["target_id"],
        type=LinkType(data["type"]),
        provenance=data["provenance"],
        created_at=data["created_at"],
    )
