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
    "progress",
    "research",
    "rule",
    "learning",
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

    # NOTE: optional params are typed `str = None` (etc.), NOT `Optional[str]`, on purpose.
    # Optional[...] renders in the tool schema as anyOf[T, null], which some MCP clients
    # surface as an "unknown" type; a bare concrete type reads cleanly. Enforced by
    # tests/integration/test_mcp_server.py::test_optional_params_expose_concrete_types —
    # do not "fix" these to Optional.

    @mcp.tool()
    def remember(
        content: Annotated[
            str,
            Field(description="The memory text. Be specific: problem, solution, reasoning. Markdown is fine."),
        ],
        type: Annotated[
            MemoryTypeName,
            Field(description="Kind of memory (shapes retrieval): decision, progress, research, rule, learning, working-notes."),
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
        that key. Returns {id, status}, where status is "created", "duplicate"
        (identical content already stored), or "superseded" (a topic_key upsert).
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
    def recall(
        query: Annotated[
            str,
            Field(description="The question to answer from this project's memory."),
        ],
        project: Annotated[
            str,
            Field(description="Project slug to recall from."),
        ],
    ) -> dict:
        """Recall a project's memory as a synthesized, grounded answer.

        Retrieves the memories most relevant to your query, then a local LLM writes an
        answer using ONLY those memories — never outside knowledge — and replies exactly
        "No relevant memories found." when none apply. Unlike `search` (a ranked list of
        hits), this returns a written answer. Returns {project, summary, sources}: `summary`
        is the answer and `sources` are the memories it drew on ({id, type}, not their
        content — so the answer stays light on the caller's context).
        """
        bundle = container.recall.execute(project=project, query=query)
        return {
            "project": bundle.project,
            "summary": bundle.summary,
            "sources": [
                {"id": memory.id, "type": section.type}
                for section in bundle.sections
                for memory in section.memories
            ],
        }

    @mcp.tool()
    def delete(
        ids: Annotated[list[str], Field(description="Ids of the memories to delete.")],
    ) -> dict:
        """Permanently delete specific memories. Returns {deleted}."""
        return asdict(container.delete.delete(ids))

    @mcp.tool()
    def purge() -> dict:
        """Permanently delete ALL memories. Returns {deleted}."""
        return asdict(container.delete.purge())

    @mcp.tool()
    def create_project(
        name: Annotated[
            str,
            Field(description="Project slug in kebab-case — the id, reused on every memory."),
        ],
        description: Annotated[
            str,
            Field(description="What this project is (optional)."),
        ] = None,
    ) -> dict:
        """Register a new project. Writing to or searching an unregistered project is
        rejected (with near-match suggestions), so create it first. Errors if the slug
        already exists. Returns {slug, description, created_at}.
        """
        return asdict(container.create_project.execute(name, description))

    @mcp.tool()
    def delete_project(
        name: Annotated[
            str,
            Field(description="Project slug to delete, with ALL its memories."),
        ],
    ) -> dict:
        """Permanently delete a project and EVERYTHING in it — all its memories and
        their links, in one atomic cascade. Errors (with near-match suggestions) if the
        slug is unknown. Returns the deleted {slug, description, created_at}.
        """
        return asdict(container.delete_project.execute(name))

    @mcp.tool()
    def update_project(
        name: Annotated[str, Field(description="Project slug to update.")],
        description: Annotated[
            str,
            Field(description="New description for the project (what it is)."),
        ],
    ) -> dict:
        """Set or change a project's description (the only way to edit it). Errors (with
        near-match suggestions) if the slug is unknown. Returns {slug, description, created_at}.
        """
        return asdict(container.update_project.execute(name, description))

    @mcp.tool()
    def list_projects() -> list[dict]:
        """List the registered projects (newest first). Returns a list of
        {slug, description, created_at}; the global scope is not a project and is excluded.
        """
        return [asdict(project) for project in container.list_projects.execute()]

    return mcp
