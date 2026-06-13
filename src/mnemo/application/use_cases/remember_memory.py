"""Store a memory. No LLM on this path: exact-dup + topic_key upsert + insert."""
from __future__ import annotations

from mnemo.application.ports.embedder import EmbedderPort
from mnemo.application.ports.memory_repository import MemoryRepositoryPort
from mnemo.application.results.remember_result import RememberResult
from mnemo.domain.constants import DEFAULT_TYPE
from mnemo.domain.memory import Memory
from mnemo.domain.memory_type import MemoryType
from mnemo.domain.scope import Scope


class RememberMemory:
    def __init__(
        self, repository: MemoryRepositoryPort, embedder: EmbedderPort
    ) -> None:
        self._repository = repository
        self._embedder = embedder

    def execute(
        self,
        *,
        content: str,
        type: MemoryType | str = DEFAULT_TYPE,
        scope: Scope | str = Scope.PROJECT,
        project: str | None = None,
        related_files: list[str] | None = None,
        tags: list[str] | None = None,
        topic_key: str | None = None,
        session_id: str | None = None,
    ) -> RememberResult:
        memory = Memory.create(
            content=content,
            type=type,
            scope=scope,
            project=project,
            related_files=related_files,
            tags=tags,
            topic_key=topic_key,
            session_id=session_id,
        )

        # Exact duplicate: identical normalized content already stored — don't spawn a row.
        exact = self._repository.find_by_hash(memory.hash)
        if exact is not None:
            self._repository.register_duplicate(exact.id)
            return RememberResult(id=exact.id, dedup="exact")

        # Explicit evolution: reusing a topic_key supersedes the prior record (kept as history).
        superseded_id: str | None = None
        if memory.topic_key is not None:
            prior = self._repository.find_active_by_topic_key(
                memory.topic_key, memory.project
            )
            if prior is not None:
                self._repository.mark_superseded(prior.id)
                memory.supersedes = prior.id
                superseded_id = prior.id

        # Near-similar memories are NOT suppressed here — they coexist; the background
        # worker may merge/flag genuine duplicates later (docs/04-data-model.md).
        vector = self._embedder.encode(memory.content)
        self._repository.add(memory, vector)
        return RememberResult(id=memory.id, superseded=superseded_id)
