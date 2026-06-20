"""In-memory project registry (offline/test double).

Persists to a small JSON file when given a path, so it survives a service restart
the same way the in-memory memory store (memory.json) does — keeping the test
backend consistent with SQLite, where memories and projects share one DB. The
reserved `__global__` sentinel is always present (exempt from the gate, hidden
from listings).
"""
from __future__ import annotations

import json
from pathlib import Path

from mnemo.domain.constants import GLOBAL_PROJECT
from mnemo.domain.generators import now
from mnemo.domain.project import Project


class InMemoryProjectRepositoryImpl:
    def __init__(self, path: str | None = None) -> None:
        self._path = Path(path) if path else None
        self._projects: dict[str, Project] = {}
        self._load()
        if GLOBAL_PROJECT not in self._projects:
            self._projects[GLOBAL_PROJECT] = Project(GLOBAL_PROJECT, None, now())
            self._persist()

    def exists(self, slug: str) -> bool:
        return slug in self._projects

    def get(self, slug: str) -> Project | None:
        return self._projects.get(slug)

    def create(self, project: Project) -> None:
        self._projects[project.slug] = project
        self._persist()

    def update_description(self, slug: str, description: str | None) -> None:
        existing = self._projects.get(slug)
        if existing is not None:
            # New value, not a mutation of the stored entity (repository purity).
            self._projects[slug] = Project(
                slug=existing.slug, description=description, created_at=existing.created_at
            )
            self._persist()

    def delete(self, slug: str) -> None:
        if self._projects.pop(slug, None) is not None:
            self._persist()

    def list_all(self) -> list[Project]:
        return [
            project for slug, project in self._projects.items() if slug != GLOBAL_PROJECT
        ]

    def _persist(self) -> None:
        # Test-only persistence (the SQLite backend is the real one); plain truncate-write.
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"slug": p.slug, "description": p.description, "created_at": p.created_at}
            for p in self._projects.values()
        ]
        self._path.write_text(json.dumps(payload))

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        for row in json.loads(self._path.read_text()):
            self._projects[row["slug"]] = Project(
                slug=row["slug"], description=row["description"], created_at=row["created_at"]
            )
