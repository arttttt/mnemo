"""MCP controller: exposes the use cases as MCP tools for AI agents.

`mcp` and `pydantic.Field` are imported lazily so the core and offline tests
don't require them. The `Literal` aliases are surfaced to MCP clients as
JSON-schema enums (so agents pick valid values) and are kept in sync with the
domain enums by a test.
"""
from dataclasses import asdict
from typing import Annotated, Literal, Optional

from mnemo.infrastructure.composition import build_container
from mnemo.infrastructure.container import Container

MemoryTypeName = Literal[
    "decision",
    "debug",
    "progress",
    "feature",
    "research",
    "code-snippet",
    "rule",
    "learning",
    "discussion",
    "design",
    "working-notes",
]
StoreScope = Literal["project", "global"]
SearchScope = Literal["project", "global", "all"]


def build_mcp(container: Optional[Container] = None, **settings):
    from mcp.server.fastmcp import FastMCP  # lazy, optional dependency
    from pydantic import Field  # provided by mcp; lazy to keep imports light

    container = container or build_container()
    # `settings` (e.g. host/port) configure the transport; stdio ignores them.
    mcp = FastMCP("mnemo", **settings)

    @mcp.tool()
    def remember(
        content: Annotated[
            str,
            Field(description="The memory text. Be specific: problem, solution, reasoning. Markdown is fine."),
        ],
        type: Annotated[
            MemoryTypeName,
            Field(description="Kind of memory (shapes retrieval): decision, debug, progress, feature, research, code-snippet, rule, learning, discussion, design, working-notes."),
        ] = "working-notes",
        project: Annotated[
            str,
            Field(description="Project slug in kebab-case. Omit and set scope='global' for cross-project knowledge."),
        ] = None,
        scope: Annotated[
            StoreScope,
            Field(description="'project' = belongs to one project; 'global' = applies everywhere (rules, cross-project lessons)."),
        ] = "project",
        related_files: Annotated[
            list[str],
            Field(description="File paths this memory concerns."),
        ] = None,
        tags: Annotated[
            list[str],
            Field(description="Optional keywords for later filtering."),
        ] = None,
        topic_key: Annotated[
            str,
            Field(description="Stable key (e.g. 'auth/jwt-model') to evolve one memory over time instead of creating duplicates."),
        ] = None,
    ) -> dict:
        """Save a memory to the local store so it can be recalled later.

        No LLM runs on write. Reusing a `topic_key` supersedes the prior memory of
        that key. Returns {id, dedup, superseded}.
        """
        result = container.remember.execute(
            content=content,
            type=type,
            project=project,
            scope=scope,
            related_files=related_files,
            tags=tags,
            topic_key=topic_key,
        )
        return asdict(result)

    @mcp.tool()
    def search(
        query: Annotated[
            str,
            Field(description="Natural-language description of what you're looking for."),
        ],
        scope: Annotated[
            SearchScope,
            Field(description="'project' (default) = the given project + global, and REQUIRES the project param; 'global' = only global; 'all' = every project (cross-project)."),
        ] = "project",
        project: Annotated[
            str,
            Field(description="Project slug to scope to. Required when scope='project' (the default), and must be omitted for scope='global'/'all'."),
        ] = None,
        type: Annotated[
            MemoryTypeName,
            Field(description="Restrict results to a single memory type."),
        ] = None,
        tags: Annotated[
            list[str],
            Field(description="Keep only memories carrying ALL of these tags."),
        ] = None,
        related_files: Annotated[
            list[str],
            Field(description="Keep only memories referencing ANY of these file paths."),
        ] = None,
        created_after: Annotated[
            str,
            Field(description="Keep only memories created at or after this ISO-8601 instant (e.g. '2026-06-01' or '2026-06-01T00:00:00+00:00')."),
        ] = None,
        limit: Annotated[
            int,
            Field(ge=1, le=100, description="Maximum number of hits to return."),
        ] = 10,
    ) -> list[dict]:
        """Find memories by meaning, ranked by relevance.

        Returns a list of hits, each {id, score, type, scope, project, content,
        related_files, created_at}. Use scope='all' for cross-project search.
        Optional filters (type, tags, related_files, created_after) narrow the results.
        """
        results = container.search.execute(
            query=query,
            scope=scope,
            project=project,
            type=type,
            tags=tags,
            related_files=related_files,
            created_after=created_after,
            limit=limit,
        )
        return [asdict(result) for result in results]

    @mcp.tool()
    def browse(
        scope: Annotated[
            SearchScope,
            Field(description="'project' (default) = the given project + global, and REQUIRES the project param; 'global' = only global; 'all' = every project (cross-project)."),
        ] = "project",
        project: Annotated[
            str,
            Field(description="Project slug to scope to. Required when scope='project' (the default), and must be omitted for scope='global'/'all'."),
        ] = None,
        type: Annotated[
            MemoryTypeName,
            Field(description="Restrict results to a single memory type."),
        ] = None,
        tags: Annotated[
            list[str],
            Field(description="Keep only memories carrying ALL of these tags."),
        ] = None,
        related_files: Annotated[
            list[str],
            Field(description="Keep only memories referencing ANY of these file paths."),
        ] = None,
        created_after: Annotated[
            str,
            Field(description="Keep only memories created at or after this ISO-8601 instant (e.g. '2026-06-01' or '2026-06-01T00:00:00+00:00')."),
        ] = None,
        limit: Annotated[
            int,
            Field(ge=1, le=100, description="Maximum number of memories to return."),
        ] = 10,
    ) -> list[dict]:
        """List memories by filter, newest first — browse without a query.

        Use this for category retrieval ("all type=decision in this project") where
        a semantic query would only bias the order. No relevance ranking, so hits
        carry no score; they are ordered by recency. Each hit is {id, type, scope,
        project, content, related_files, created_at}. Use `search` to find by meaning.
        """
        results = container.browse.execute(
            scope=scope,
            project=project,
            type=type,
            tags=tags,
            related_files=related_files,
            created_after=created_after,
            limit=limit,
        )
        return [asdict(result) for result in results]

    @mcp.tool()
    def delete(
        ids: Annotated[list[str], Field(description="Ids of the memories to delete.")],
    ) -> dict:
        """Permanently delete specific memories. Returns {deleted}."""
        return asdict(container.delete.delete(ids))

    @mcp.tool()
    def clear(
        project: Annotated[str, Field(description="Project whose memories to delete.")],
    ) -> dict:
        """Permanently delete all memories of one project. Returns {deleted}."""
        return asdict(container.delete.clear(project))

    @mcp.tool()
    def purge() -> dict:
        """Permanently delete ALL memories. Returns {deleted}."""
        return asdict(container.delete.purge())

    return mcp
