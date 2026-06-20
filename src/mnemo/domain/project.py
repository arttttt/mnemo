"""The Project entity — the `project` column promoted to a first-class, registered
value. `slug` IS the id (no surrogate); it stays the key on memory rows."""
from __future__ import annotations

from dataclasses import dataclass

from mnemo.domain.generators import now


@dataclass
class Project:
    slug: str
    description: str | None
    created_at: str

    @classmethod
    def create(cls, slug: str, description: str | None = None) -> "Project":
        if not slug or not slug.strip():
            raise ValueError("project slug is empty")
        return cls(slug=slug, description=description, created_at=now())
