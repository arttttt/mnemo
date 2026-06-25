"""Result of dereferencing one memory by id or topic_key.

The full resolved record plus a LIGHT index of its supersede chain (the version
lineage): chain entries carry no content — only {id, status, created_at} — so a long
history stays cheap; fetch a specific older version with another `get(id=...)`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChainEntry:
    """One version in a memory's supersede lineage — light (no content)."""

    id: str
    status: str
    created_at: str


@dataclass(frozen=True)
class GetResult:
    id: str
    type: str
    scope: str
    project: str | None
    content: str
    related_files: list[str]
    created_at: str
    topic_key: str | None
    status: str
    supersedes: str | None
    chain: list[ChainEntry]   # the version lineage, newest -> oldest, capped at chain_limit
    chain_total: int          # total versions in the lineage (page older ones with chain_after)
