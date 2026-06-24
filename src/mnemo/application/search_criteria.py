"""Structured retrieval criteria — what to keep, independent of the backend.

A value object (specification): it carries the filters and knows how to test a
memory in-process. Each store also translates it to its own query (e.g. a SQL
WHERE) for pushed-down filtering. Active memories by default; `status` widens to
superseded or all.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

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
    status: str = "active"                  # 'active' (default) | 'superseded' | 'all'

    def __post_init__(self) -> None:
        # Parse created_after to a datetime and normalize it to UTC up front: a malformed
        # value would otherwise filter silently and wrongly, and a non-UTC offset would
        # mis-order under the string comparison the SQL store uses. Stored created_at is
        # always UTC ISO (domain.now()), so a UTC-normalized bound compares correctly in
        # both the in-process matcher and the SQL `>=`.
        if self.created_after is not None:
            try:
                parsed = datetime.fromisoformat(self.created_after.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError(
                    "created_after must be an ISO-8601 date or datetime (e.g. "
                    f"'2026-06-01' or '2026-06-01T00:00:00+00:00'); got {self.created_after!r}"
                )
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)  # naive input is taken as UTC
            object.__setattr__(self, "created_after", parsed.astimezone(timezone.utc).isoformat())
        # The scope↔project contract, shared with browse (one source of truth).
        validate_scope_project(self.scope, self.project)
        if self.status not in ("active", "superseded", "all"):
            raise ValueError(
                f"status must be 'active', 'superseded' or 'all'; got {self.status!r}"
            )

    def matches(self, memory: Memory) -> bool:
        if self.status == "active" and memory.status != "active":
            return False
        if self.status == "superseded" and memory.status != "superseded":
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
        if self.created_after is not None and (
            datetime.fromisoformat(memory.created_at) < datetime.fromisoformat(self.created_after)
        ):
            return False
        return True

    def _in_scope(self, memory: Memory) -> bool:
        if self.scope == "all":
            return True
        if self.scope == "global":
            return memory.scope is Scope.GLOBAL
        return memory.project == self.project or memory.scope is Scope.GLOBAL
