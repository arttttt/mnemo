"""Structured retrieval criteria — what to keep, independent of the backend.

A value object (specification): it carries the filters and knows how to test a
memory in-process. Each store also translates it to its own query (e.g. a SQL
WHERE) for pushed-down filtering. Only active memories are ever returned.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from mnemo.application.scope_contract import validate_scope_project
from mnemo.domain.memory import Memory
from mnemo.domain.memory_type import MemoryType
from mnemo.domain.scope import Scope


@dataclass(frozen=True)
class SearchCriteria:
    scope: str = "project"                  # 'project' (this + global) | 'global' | 'all'
    project: str | None = None
    type: MemoryType | None = None
    tags: tuple[str, ...] = ()              # memory must carry ALL of these
    related_files: tuple[str, ...] = ()     # memory must reference ANY of these
    created_after: str | None = None        # ISO-8601 lower bound; keep created_at >= this

    def __post_init__(self) -> None:
        # created_after is compared lexicographically against the stored ISO created_at,
        # so a malformed value would filter silently and wrongly — validate it as ISO-8601
        # (date or datetime) up front and reject anything else with a clear error.
        if self.created_after is not None:
            try:
                datetime.fromisoformat(self.created_after.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError(
                    "created_after must be an ISO-8601 date or datetime (e.g. "
                    f"'2026-06-01' or '2026-06-01T00:00:00+00:00'); got {self.created_after!r}"
                )
        # The scope↔project contract, shared with browse and clear (one source of truth).
        validate_scope_project(self.scope, self.project)

    def matches(self, memory: Memory) -> bool:
        if memory.status != "active":
            return False
        if not self._in_scope(memory):
            return False
        if self.type is not None and memory.type != self.type:
            return False
        if self.tags and not all(tag in memory.tags for tag in self.tags):
            return False
        if self.related_files and not any(
            path in memory.related_files for path in self.related_files
        ):
            return False
        if self.created_after is not None and memory.created_at < self.created_after:
            return False
        return True

    def _in_scope(self, memory: Memory) -> bool:
        if self.scope == "all":
            return True
        if self.scope == "global":
            return memory.scope is Scope.GLOBAL
        return memory.project == self.project or memory.scope is Scope.GLOBAL
