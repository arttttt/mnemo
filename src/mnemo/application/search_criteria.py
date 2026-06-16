"""Structured retrieval criteria — what to keep, independent of the backend.

A value object (specification): it carries the filters and knows how to test a
memory in-process. Each store also translates it to its own query (e.g. a SQL
WHERE) for pushed-down filtering. Only active memories are ever returned.
"""
from __future__ import annotations

from dataclasses import dataclass

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
    created_after: str | None = None        # ISO timestamp; keep created_at >= this

    def __post_init__(self) -> None:
        # A project-scoped search must name the project to scope to. There is no
        # "current project" to infer here — without a project the search would
        # silently match only project-less + global rows (almost always nothing),
        # so reject it with an explicit, actionable error instead.
        if self.scope == "project" and self.project is None:
            raise ValueError(
                "scope='project' needs a project to scope to, but none was given; "
                "pass the project slug, or use scope='global' (only global memories) "
                "or scope='all' (search across every project)"
            )
        # The mirror case: 'all'/'global' ignore project entirely (scope is
        # authoritative). Accepting a project here would silently drop the filter and
        # return a wrong-scoped result, so reject the contradiction loudly instead.
        if self.scope in ("global", "all") and self.project is not None:
            raise ValueError(
                f"project has no effect with scope='{self.scope}'; drop the project "
                f"param, or use scope='project' to scope the search to it"
            )

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
