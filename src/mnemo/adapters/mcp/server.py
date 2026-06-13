"""MCP controller: exposes the use cases as MCP tools for AI agents.

`mcp` is imported lazily so the core and offline tests don't require it.
"""
from __future__ import annotations

from dataclasses import asdict

from mnemo.infrastructure.container import Container, build_container


def build_mcp(container: Container | None = None):
    from mcp.server.fastmcp import FastMCP  # lazy, optional dependency

    container = container or build_container()
    mcp = FastMCP("mnemo")

    @mcp.tool()
    def remember(
        content: str,
        type: str = "working-notes",
        project: str | None = None,
        scope: str = "project",
        related_files: list[str] | None = None,
        tags: list[str] | None = None,
        importance: float = 0.5,
        topic_key: str | None = None,
    ) -> dict:
        """Store a memory. No LLM on this path; type/scope are parameters."""
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
        query: str,
        scope: str = "project",
        project: str | None = None,
        type: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search memory. scope: project (default, +global) | global | all (cross-project)."""
        results = container.search.execute(
            query=query, scope=scope, project=project, type=type, limit=limit
        )
        return [asdict(result) for result in results]

    return mcp


def main() -> None:
    build_mcp().run()
