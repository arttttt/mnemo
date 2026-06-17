"""A memory the pipeline proposes to create — the merged, summarized or insight record.

It mirrors the fields the write path needs; a separate executor turns it into a real
``Memory`` when the plan is applied.
"""
from __future__ import annotations

from dataclasses import dataclass

from mnemo.domain.memory_type import MemoryType
from mnemo.domain.scope import Scope


@dataclass(frozen=True)
class ProposedMemory:
    content: str
    type: MemoryType
    project: str | None
    scope: Scope = Scope.PROJECT
    related_files: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
