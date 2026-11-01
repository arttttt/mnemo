"""Build a real temp-file SQLite store for integration/contract tests.

Mirrors the composition root: the project registry is built FIRST (so the
`projects` table exists before the memories schema that foreign-keys to it), then
the memory store, both sharing one connection. The given project slugs are
registered so memory writes satisfy the `memories.project -> projects(slug)` FK —
registration is idempotent, so a test may reopen the same path repeatedly.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest


def open_store(tmp_path, dim, projects=()):
    """Return ``(memory_repo, project_repo)`` over a temp-file SQLite DB with the
    given projects registered."""
    return _build(_connections(tmp_path), dim, projects)


@contextmanager
def opened_store(tmp_path, dim, projects=()):
    """``open_store`` whose connections are closed on exit — for a DB another process
    (e.g. a running service) also holds open."""
    conns = _connections(tmp_path)
    try:
        yield _build(conns, dim, projects)
    finally:
        conns.close()


def _connections(tmp_path):
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.sqlite_connections import SqliteConnections

    return SqliteConnections(str(tmp_path / "memory.db"))


def _build(conns, dim, projects):
    from mnemo.adapters.store.sqlite_project_repository import (
        SqliteProjectRepositoryImpl,
    )
    from mnemo.adapters.store.sqlite_vec_repository import SqliteRepositoryImpl
    from mnemo.domain.project import Project

    registry = SqliteProjectRepositoryImpl(conns)  # creates `projects` first
    for slug in projects:
        if not registry.exists(slug):
            registry.create(Project.create(slug))
    return SqliteRepositoryImpl(conns, dim), registry
