"""MCP controller: exposes the use cases as MCP tools for AI agents.

`mcp` and `pydantic.Field` are imported lazily so the core and offline tests
don't require them. The `Literal` aliases below are surfaced to MCP clients as
JSON-schema enums (so agents pick valid values instead of guessing) and are kept
in sync with the domain enums by a test.
"""

from dataclasses import asdict
from typing import Annotated, Literal, Optional

from mnemo.infrastructure.container import Container, build_container

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


def build_mcp(container: Optional[Container] = None):
    from mcp.server.fastmcp import FastMCP  # lazy, optional dependency
    from pydantic import Field  # provided by mcp; lazy to keep imports light

    container = container or build_container()
    mcp = FastMCP("mnemo")

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
            Optional[str],
            Field(description="Project slug in kebab-case this memory belongs to. Omit and set scope='global' for cross-project knowledge."),
        ] = None,
        scope: Annotated[
            StoreScope,
            Field(description="'project' = belongs to one project; 'global' = applies everywhere (rules, cross-project lessons)."),
        ] = "project",
        related_files: Annotated[
            Optional[list[str]],
            Field(description="File paths this memory concerns."),
        ] = None,
        tags: Annotated[
            Optional[list[str]],
            Field(description="Optional keywords for later filtering."),
        ] = None,
        importance: Annotated[
            float,
            Field(ge=0.0, le=1.0, description="0.0-1.0 (0.9 critical, 0.7 important, 0.5 normal, 0.3 minor). Optional; caller-set, defaults to 0.5. Automatic scoring is planned."),
        ] = 0.5,
        topic_key: Annotated[
            Optional[str],
            Field(description="Stable key (e.g. 'auth/jwt-model') to evolve one memory over time instead of creating duplicates."),
        ] = None,
    ) -> dict:
        """Save a memory to the local store so it can be recalled later.

        No LLM runs on write. Call this after a meaningful decision, bug fix,
        learning, or progress checkpoint. Returns {id, dedup, score}: `dedup` is
        null for a new memory, or "exact"/"near" if it matched an existing one.
        """
        result = container.remember.execute(
            content=content,
            type=type,
            project=project,
            scope=scope,
            related_files=related_files,
            tags=tags,
            importance=importance,
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
            Field(description="'project' (default) = current project + global; 'global' = only global; 'all' = every project (cross-project)."),
        ] = "project",
        project: Annotated[
            Optional[str],
            Field(description="Project slug to scope to when scope='project'."),
        ] = None,
        type: Annotated[
            Optional[MemoryTypeName],
            Field(description="Restrict results to a single memory type."),
        ] = None,
        limit: Annotated[
            int,
            Field(ge=1, le=100, description="Maximum number of hits to return."),
        ] = 10,
    ) -> list[dict]:
        """Find memories by meaning, ranked by relevance.

        Returns a list of hits, each {id, score, type, scope, project, content,
        related_files, created_at}. Use scope='all' for cross-project search.
        """
        results = container.search.execute(
            query=query, scope=scope, project=project, type=type, limit=limit
        )
        return [asdict(result) for result in results]

    return mcp


def main() -> None:
    build_mcp().run()
