"""Builds the predicate that implements soft scoping for retrieval."""
from __future__ import annotations

from mnemo.application.types import MemoryPredicate
from mnemo.domain.memory import Memory
from mnemo.domain.memory_type import MemoryType
from mnemo.domain.scope import Scope


def scope_predicate(
    *, scope: str, project: str | None, type_filter: str | None
) -> MemoryPredicate:
    """Soft scoping: 'project' = this project OR global; 'all' = cross-project."""
    wanted_type = MemoryType(type_filter) if type_filter else None

    def predicate(memory: Memory) -> bool:
        if memory.status != "active":
            return False
        if wanted_type is not None and memory.type != wanted_type:
            return False
        if scope == "all":
            return True
        if scope == "global":
            return memory.scope is Scope.GLOBAL
        return memory.project == project or memory.scope is Scope.GLOBAL

    return predicate
