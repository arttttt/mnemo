"""Dereference ONE memory by id or topic_key — the full record plus its supersede chain.

Exact and deterministic (an indexed point lookup, no LLM, no ranking). `id` and `topic_key`
are MUTUALLY EXCLUSIVE — `id` is the stronger, global key (the exact record, any status);
`topic_key` resolves the chain's ACTIVE head within a project/global scope. A miss is a loud
error (with near-match suggestions for a topic_key), mirroring the project gate — never a
silent empty, because the caller is dereferencing a handle it believes exists.
"""
from __future__ import annotations

import difflib

from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.project_gate import ProjectGate
from mnemo.application.results.get_result import ChainEntry, GetResult
from mnemo.application.scope_contract import validate_scope_project
from mnemo.domain.constants import GLOBAL_PROJECT
from mnemo.domain.memory import Memory

_MAX_SUGGESTIONS = 5


class GetMemoryUseCaseImpl:
    def __init__(self, repository: MemoryRepository, gate: ProjectGate) -> None:
        self._repository = repository
        self._gate = gate

    def execute(
        self,
        *,
        id: str | None = None,
        topic_key: str | None = None,
        project: str | None = None,
        scope: str = "project",
        chain_limit: int = 10,
        chain_after: str | None = None,
    ) -> GetResult:
        if (id is None) == (topic_key is None):
            raise ValueError(
                "pass exactly one of id or topic_key — they are mutually exclusive "
                "(id is the exact record; topic_key resolves a chain's active head)"
            )
        resolved = (
            self._resolve_by_id(id)
            if id is not None
            else self._resolve_by_topic_key(topic_key, scope, project)
        )
        chain, chain_total = self._chain_of(resolved, chain_limit, chain_after)
        return GetResult(
            id=resolved.id,
            type=resolved.type.value,
            scope=resolved.scope.value,
            project=resolved.project,
            content=resolved.content,
            related_files=resolved.related_files,
            created_at=resolved.created_at,
            topic_key=resolved.topic_key,
            status=resolved.status,
            supersedes=resolved.supersedes,
            chain=chain,
            chain_total=chain_total,
        )

    def _resolve_by_id(self, memory_id: str) -> Memory:
        memory = self._repository.find_by_id(memory_id)
        if memory is None:
            # Ids are opaque, so there is no useful near-match to suggest.
            raise ValueError(f"no memory with id {memory_id!r}")
        return memory

    def _resolve_by_topic_key(
        self, topic_key: str, scope: str, project: str | None
    ) -> Memory:
        # A topic_key is scoped to one project or to global — 'all' is for cross-project
        # search, not an exact dereference.
        if scope not in ("project", "global"):
            raise ValueError(
                "get by topic_key needs scope 'project' or 'global' (not 'all'); a "
                "topic_key lives in one project or in global"
            )
        # The same scope<->project contract + registry gate as the other read tools.
        validate_scope_project(scope, project)
        self._gate.check(scope, project)
        column = GLOBAL_PROJECT if scope == "global" else project
        memory = self._repository.find_active_by_topic_key(topic_key, column)
        if memory is None:
            suggestions = difflib.get_close_matches(
                topic_key, self._repository.topic_keys(column),
                n=_MAX_SUGGESTIONS, cutoff=0.0,
            )
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            where = "global" if scope == "global" else f"project {project!r}"
            raise ValueError(
                f"no active memory with topic_key {topic_key!r} in {where}.{hint}"
            )
        return memory

    def _chain_of(
        self, resolved: Memory, limit: int, after_id: str | None
    ) -> tuple[list[ChainEntry], int]:
        if resolved.topic_key is None:
            # A one-off memory (no topic_key) has no lineage beyond itself.
            entry = ChainEntry(
                id=resolved.id, status=resolved.status, created_at=resolved.created_at
            )
            return [entry], 1
        chain = self._repository.chain(
            resolved.topic_key, resolved.project, limit=limit, after_id=after_id
        )
        total = self._repository.chain_length(resolved.topic_key, resolved.project)
        return chain, total
