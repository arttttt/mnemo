"""Result of a remember operation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RememberResult:
    id: str
    dedup: str | None = None        # None | "exact"
    superseded: str | None = None   # id of the record this one superseded (topic_key upsert)
