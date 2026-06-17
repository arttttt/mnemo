"""(De)serialize a Memory to/from a plain dict for JSON persistence."""
from __future__ import annotations

from mnemo.domain.memory import Memory
from mnemo.domain.memory_type import MemoryType
from mnemo.domain.scope import Scope


def to_dict(memory: Memory) -> dict:
    return {
        "id": memory.id,
        "content": memory.content,
        "type": memory.type.value,
        "scope": memory.scope.value,
        "project": memory.project,
        "related_files": memory.related_files,
        "tags": memory.tags,
        "topic_key": memory.topic_key,
        "session_id": memory.session_id,
        "status": memory.status,
        "supersedes": memory.supersedes,
        "hash": memory.hash,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
    }


def from_dict(data: dict) -> Memory:
    return Memory(
        id=data["id"],
        content=data["content"],
        type=MemoryType(data["type"]),
        scope=Scope(data["scope"]),
        project=data["project"],
        related_files=data["related_files"],
        tags=data["tags"],
        topic_key=data["topic_key"],
        session_id=data["session_id"],
        status=data["status"],
        supersedes=data["supersedes"],
        hash=data["hash"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )
