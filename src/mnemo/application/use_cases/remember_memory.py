"""Store a memory. No LLM on this path: exact-dup + topic_key upsert + insert.

The embedding is NOT computed here — the memory is inserted pending and handed to
the embedding scheduler (sync inline, or deferred to a background worker). See
docs/03-architecture.md (deferred embedding).
"""
from __future__ import annotations

from mnemo.application.ports.embedding_scheduler import EmbeddingScheduler
from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.ports.session_provider import SessionProvider
from mnemo.application.ports.token_window import TokenWindow
from mnemo.application.project_gate import ProjectGate
from mnemo.application.results.remember_result import RememberResult
from mnemo.application.scope_contract import validate_scope_project
from mnemo.application.token_budget import TokenBudget
from mnemo.domain.constants import DEFAULT_TYPE
from mnemo.domain.memory import Memory
from mnemo.domain.memory_type import MemoryType
from mnemo.domain.scope import Scope


class RememberMemoryUseCaseImpl:
    def __init__(
        self,
        repository: MemoryRepository,
        scheduler: EmbeddingScheduler,
        token_window: TokenWindow,
        session_provider: SessionProvider,
        gate: ProjectGate,
    ) -> None:
        self._repository = repository
        self._scheduler = scheduler
        self._token_window = token_window
        self._budget = TokenBudget(token_window)
        self._session_provider = session_provider
        self._gate = gate

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
        # A project-scoped write must target a REGISTERED project (else a typo'd slug
        # would create an invisible phantom project). global/all are exempt.
        self._gate.check(scope, project)
        memory = Memory.create(
            content=content,
            type=type,
            scope=scope,
            project=project,
            related_files=related_files,
            tags=tags,
            topic_key=topic_key,
        )

        # Length guard: keep a memory within its TYPE's cap (a rule is far tighter than the rest)
        # AND inside the embedder's window (a memory is one vector) — the effective limit is the
        # stricter of the two. Reject with an explicit error (never truncate, never auto-split) so
        # the caller — already an LLM — can resolve it deliberately: tighten the wording to fit
        # first (one focused memory beats many fragments), and split only when it genuinely can't
        # be condensed. The token count is cheap and stays on the hot path; only the encode is
        # deferred.
        self._budget.ensure_within(
            memory.content,
            min(memory.type.max_tokens, self._token_window.max_input),
            subject="content",
            qualifier=f" for a '{memory.type.value}' memory",
            advice="tighten the wording to be more concise and precise so it fits, and split it "
            "into smaller, focused memories only if it genuinely can't be condensed",
        )

        # Exact duplicate: identical normalized content already ACTIVE in this same
        # scope/project — don't spawn a row. The lookup is project-scoped (the same
        # content is a distinct memory in another project) and active-only (re-storing
        # previously superseded content writes a fresh, retrievable row).
        exact = self._repository.find_active_by_hash(memory.hash, memory.project)
        if exact is not None:
            # Re-storing identical content under a *new* topic_key would otherwise be
            # silently dropped here — this guard runs BEFORE the topic_key upsert below, so
            # the caller's intended evolution never happens. Make it a loud error instead.
            # (Editing/re-keying identical content in place is a deliberate post-MVP op —
            # see docs/roadmap/post-mvp.md.) No topic_key, or the same one, is an idempotent
            # re-store → the existing soft "duplicate".
            if memory.topic_key is not None and memory.topic_key != exact.topic_key:
                raise ValueError(
                    f"identical content is already stored as memory {exact.id} "
                    f"(topic_key={exact.topic_key!r}); storing it again under a different "
                    f"topic_key ({memory.topic_key!r}) is not supported — change the content "
                    f"to evolve the memory, or delete {exact.id} first to re-key it"
                )
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
            # supersedes the prior (recorded in the `supersedes` column). The repository
            # persists the mark + insert in one transaction, so a crash can never strand
            # the topic_key with no active record.
            memory.supersedes = prior.id
            self._repository.supersede(memory)
            status = "superseded"
        else:
            self._repository.add(memory)
            status = "created"

        # Hand the embedding off only AFTER the write has committed, so the background
        # worker (reading via its own connection) is guaranteed to see the pending row.
        self._scheduler.schedule(memory.id)
        return RememberResult(id=memory.id, status=status)
