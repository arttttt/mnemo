"""The Memory entity. Build it via :meth:`create` to enforce invariants."""
from __future__ import annotations

from dataclasses import dataclass

from mnemo.domain.constants import DEFAULT_TYPE, GLOBAL_PROJECT
from mnemo.domain.generators import new_id, now
from mnemo.domain.hashing import content_hash
from mnemo.domain.memory_type import MemoryType
from mnemo.domain.scope import Scope


@dataclass
class Memory:
    id: str
    content: str
    type: MemoryType
    scope: Scope
    project: str | None
    related_files: list[str]
    tags: list[str]
    topic_key: str | None
    session_id: str | None
    status: str
    supersedes: str | None
    hash: str
    created_at: str
    updated_at: str
    last_seen_at: str
    duplicate_count: int

    @classmethod
    def create(
        cls,
        content: str,
        *,
        type: MemoryType | str = DEFAULT_TYPE,
        scope: Scope | str = Scope.PROJECT,
        project: str | None = None,
        related_files: list[str] | None = None,
        tags: list[str] | None = None,
        topic_key: str | None = None,
        session_id: str | None = None,
        id: str | None = None,
    ) -> "Memory":
        scope = Scope(scope)
        type = MemoryType(type)
        if scope is Scope.GLOBAL:
            project = GLOBAL_PROJECT
        stamp = now()
        return cls(
            id=id or new_id(),
            content=content,
            type=type,
            scope=scope,
            project=project,
            related_files=list(related_files or []),
            tags=list(tags or []),
            topic_key=topic_key,
            session_id=session_id,
            status="active",
            supersedes=None,
            hash=content_hash(content),
            created_at=stamp,
            updated_at=stamp,
            last_seen_at=stamp,
            duplicate_count=0,
        )

    def register_duplicate(self) -> None:
        """Record that an identical memory was seen again."""
        self.duplicate_count += 1
        self.last_seen_at = now()

    def mark_superseded(self) -> None:
        """Mark this record as replaced by a newer one (kept for history)."""
        self.status = "superseded"
        self.updated_at = now()
