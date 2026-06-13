"""Domain layer: the Memory entity and pure business rules.

No framework or I/O dependencies (NFR-20/21) — only the standard library.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

GLOBAL_PROJECT = "__global__"


class MemoryType(str, Enum):
    DECISION = "decision"
    DEBUG = "debug"
    PROGRESS = "progress"
    FEATURE = "feature"
    RESEARCH = "research"
    CODE_SNIPPET = "code-snippet"
    RULE = "rule"
    LEARNING = "learning"
    DISCUSSION = "discussion"
    DESIGN = "design"
    WORKING_NOTES = "working-notes"


class Scope(str, Enum):
    PROJECT = "project"
    GLOBAL = "global"


DEFAULT_TYPE = MemoryType.WORKING_NOTES


def normalize(text: str) -> str:
    """Canonical form for hashing/dedup: collapsed whitespace, lowercased."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def content_hash(text: str) -> str:
    """Stable content fingerprint used for exact dedup."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Memory:
    """A single stored memory. Build it via :meth:`create` to enforce invariants."""

    id: str
    content: str
    type: MemoryType
    scope: Scope
    project: str | None
    related_files: list[str]
    tags: list[str]
    importance: float
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
        importance: float = 0.5,
        topic_key: str | None = None,
        session_id: str | None = None,
        id: str | None = None,
    ) -> "Memory":
        scope = Scope(scope)
        type = MemoryType(type)
        if scope is Scope.GLOBAL:
            project = GLOBAL_PROJECT
        now = _now()
        return cls(
            id=id or _new_id(),
            content=content,
            type=type,
            scope=scope,
            project=project,
            related_files=list(related_files or []),
            tags=list(tags or []),
            importance=float(importance),
            topic_key=topic_key,
            session_id=session_id,
            status="active",
            supersedes=None,
            hash=content_hash(content),
            created_at=now,
            updated_at=now,
            last_seen_at=now,
            duplicate_count=0,
        )

    def register_duplicate(self) -> None:
        """Record that an identical memory was seen again."""
        self.duplicate_count += 1
        self.last_seen_at = _now()
