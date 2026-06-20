"""Build a real temp-file SQLite store for integration/contract tests.

Mirrors the composition root: the project registry is built FIRST (so the
`projects` table exists before the memories schema that foreign-keys to it), then
the memory store, both sharing one connection. The given project slugs are
registered so memory writes satisfy the `memories.project -> projects(slug)` FK —
registration is idempotent, so a test may reopen the same path repeatedly.
"""
from __future__ import annotations

import pytest


def open_store(tmp_path, dim, projects=()):
    """Return ``(memory_repo, project_repo)`` over a temp-file SQLite DB with the
    given projects registered."""
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.sqlite_connections import SqliteConnections
    from mnemo.adapters.store.sqlite_project_repository import (
        SqliteProjectRepositoryImpl,
    )
    from mnemo.adapters.store.sqlite_vec_repository import SqliteRepositoryImpl
    from mnemo.domain.project import Project

    conns = SqliteConnections(str(tmp_path / "memory.db"))
    registry = SqliteProjectRepositoryImpl(conns)  # creates `projects` first
    for slug in projects:
        if not registry.exists(slug):
            registry.create(Project.create(slug))
    return SqliteRepositoryImpl(conns, dim), registry
