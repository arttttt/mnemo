"""Store a memory. No LLM on this path: exact-dup + topic_key upsert + insert.

The embedding is NOT computed here — the memory is inserted pending and handed to
the embedding scheduler (sync inline, or deferred to a background worker). See
docs/03-architecture.md (deferred embedding).
"""
from __future__ import annotations

from mnemo.application.ports.embedding_scheduler import EmbeddingSchedulerPort
from mnemo.application.ports.memory_repository import MemoryRepositoryPort
from mnemo.application.ports.session_provider import SessionProviderPort
from mnemo.application.results.remember_result import RememberResult
from mnemo.application.scope_contract import validate_scope_project
from mnemo.domain.constants import DEFAULT_TYPE
from mnemo.domain.link import Link
from mnemo.domain.memory import Memory
from mnemo.domain.memory_type import MemoryType
from mnemo.domain.scope import Scope


class RememberMemory:
    def __init__(
        self,
        repository: MemoryRepositoryPort,
        scheduler: EmbeddingSchedulerPort,
        session_provider: SessionProviderPort,
    ) -> None:
        self._repository = repository
        self._scheduler = scheduler
        self._session_provider = session_provider

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
    ) -> RememberResult:
        # Same scope↔project contract the read path enforces: a project-scoped write must
        # name its project (else it is silently unreachable by a project search), and a
        # global write must not carry one (scope is authoritative).
        validate_scope_project(scope, project)
        memory = Memory.create(
            content=content,
            type=type,
            scope=scope,
            project=project,
            related_files=related_files,
            tags=tags,
            topic_key=topic_key,
        )

        # Over-window guard: a memory is one vector, so it must fit the embedder's
        # token window. Reject with an explicit error (never truncate, never auto-split)
        # so the caller — already an LLM — can split it deliberately. The token count is
        # cheap and stays on the hot path; only the encode is deferred.
        tokens = self._scheduler.count_tokens(memory.content)
        if tokens > self._scheduler.max_input:
            raise ValueError(
                f"content is {tokens} tokens, over the embedder's window of "
                f"{self._scheduler.max_input}; split it into smaller, focused memories"
            )

        # Exact duplicate: identical normalized content already ACTIVE in this same
        # scope/project — don't spawn a row. The lookup is project-scoped (the same
        # content is a distinct memory in another project) and active-only (re-storing
        # previously superseded content writes a fresh, retrievable row).
        exact = self._repository.find_active_by_hash(memory.hash, memory.project)
        if exact is not None:
            return RememberResult(id=exact.id, status="duplicate")

        # Explicit evolution: reusing a topic_key supersedes the prior active record.
        prior = None
        if memory.topic_key is not None:
            prior = self._repository.find_active_by_topic_key(
                memory.topic_key, memory.project
            )

        # Provenance: stamp the current run's session id. Only stored memories get one,
        # so a read-only run generates nothing. The agent never sets it.
        memory.session_id = self._session_provider.current_session_id()

        # Insert PENDING (no vector) so the write stays cheap and the row is lexically
        # searchable at once. Near-similar memories are NOT suppressed here — they
        # coexist; the background worker may merge/flag genuine duplicates later
        # (docs/04-data-model.md).
        if prior is not None:
            # Establish the relationship HERE (application layer owns it): the successor
            # supersedes the prior, and the typed edge records it with provenance = the
            # topic_key that drove the upsert. The repository then persists all of it in
            # one transaction, so a crash can never strand the topic_key or drop the edge.
            memory.supersedes = prior.id
            link = Link.supersedes(
                source_id=memory.id, target_id=prior.id, provenance=memory.topic_key
            )
            self._repository.supersede(memory, link)
            status = "superseded"
        else:
            self._repository.add(memory)
            status = "created"

        # Hand the embedding off only AFTER the write has committed, so the background
        # worker (reading via its own connection) is guaranteed to see the pending row.
        self._scheduler.schedule(memory.id)
        return RememberResult(id=memory.id, status=status)
