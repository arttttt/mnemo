"""SQLite project registry — owns the `projects` table.

Shares the memory store's SqliteConnections (one DB, one writer, one lock) so the
FK ON DELETE CASCADE (delete a project -> its memories -> their links) runs
atomically on a single connection. The reserved `__global__` row is seeded so
global memories (which carry project='__global__') satisfy that FK; it is exempt
from the gate and hidden from listings — global is a scope, not a project.
"""
from __future__ import annotations

import sqlite3

from mnemo.adapters.store.executors import SqlReadExecutor, SqlWriteExecutor
from mnemo.adapters.store.sqlite_connections import SqliteConnections
from mnemo.domain.constants import GLOBAL_PROJECT
from mnemo.domain.generators import now
from mnemo.domain.project import Project


class SqliteProjectRepositoryImpl:
    def __init__(self, connections: SqliteConnections) -> None:
        self._read = SqlReadExecutor(connections)
        self._write = SqlWriteExecutor(connections)
        self._write.execute(self._ensure_schema)

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS projects ("
            " slug TEXT PRIMARY KEY, description TEXT, created_at TEXT NOT NULL)"
        )
        # Reserved system row for the FK integrity of global memories — not a real
        # project (excluded from list_all, exempt from the gate).
        conn.execute(
            "INSERT OR IGNORE INTO projects (slug, description, created_at) VALUES (?, NULL, ?)",
            (GLOBAL_PROJECT, now()),
        )

    def exists(self, slug: str) -> bool:
        row = self._read.execute(
            lambda conn: conn.execute(
                "SELECT 1 FROM projects WHERE slug = ?", (slug,)
            ).fetchone()
        )
        return row is not None

    def get(self, slug: str) -> Project | None:
        row = self._read.execute(
            lambda conn: conn.execute(
                "SELECT slug, description, created_at FROM projects WHERE slug = ?", (slug,)
            ).fetchone()
        )
        return self._to_project(row) if row else None

    def create(self, project: Project) -> None:
        self._write.execute(
            lambda conn: conn.execute(
                "INSERT INTO projects (slug, description, created_at) VALUES (?, ?, ?)",
                (project.slug, project.description, project.created_at),
            )
        )

    def update_description(self, slug: str, description: str | None) -> None:
        self._write.execute(
            lambda conn: conn.execute(
                "UPDATE projects SET description = ? WHERE slug = ?", (description, slug)
            )
        )

    def delete(self, slug: str) -> None:
        self._write.execute(
            lambda conn: conn.execute("DELETE FROM projects WHERE slug = ?", (slug,))
        )

    def list_all(self) -> list[Project]:
        rows = self._read.execute(
            lambda conn: conn.execute(
                "SELECT slug, description, created_at FROM projects"
                " WHERE slug != ? ORDER BY created_at DESC",
                (GLOBAL_PROJECT,),
            ).fetchall()
        )
        return [self._to_project(row) for row in rows]

    @staticmethod
    def _to_project(row) -> Project:
        return Project(slug=row[0], description=row[1], created_at=row[2])
