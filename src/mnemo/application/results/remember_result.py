"""Result of a remember operation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RememberResult:
    id: str
    # "created" (new row) | "duplicate" (exact content already stored, nothing written)
    # | "superseded" (a topic_key upsert replaced a prior memory; the edge lives in `links`)
    status: str
