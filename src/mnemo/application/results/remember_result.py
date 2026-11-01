"""Result of a remember operation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReplacedMemory:
    """The record a supersede replaced: its id, and its length in characters, so the
    writer can tell whether it just dropped more than it wrote."""

    id: str
    chars: int


@dataclass(frozen=True)
class RememberResult:
    id: str
    # "created" (new row) | "duplicate" (exact content already stored, nothing written)
    # | "superseded" (a topic_key upsert replaced a prior memory, now in its `supersedes`)
    status: str
    # Only for "superseded": the record that was replaced and left search and browse.
    replaced: ReplacedMemory | None = None
