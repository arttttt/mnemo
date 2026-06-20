"""The Project entity — the `project` column promoted to a first-class, registered
value. `slug` IS the id (no surrogate); it stays the key on memory rows."""
from __future__ import annotations

import re
from dataclasses import dataclass

from mnemo.domain.generators import now

# kebab-case: lowercase letters/digits in hyphen-separated groups — no spaces, no
# uppercase, no leading/trailing/double hyphens, no other punctuation. The slug is
# the id reused on every memory, so it is validated at creation (not normalized:
# malformed input is rejected, never silently transformed). The reserved
# `__global__` sentinel is seeded directly, bypassing this path.
_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass
class Project:
    slug: str
    description: str | None
    created_at: str

    @classmethod
    def create(cls, slug: str, description: str | None = None) -> "Project":
        if not slug or not slug.strip():
            raise ValueError("project slug is empty")
        if not _SLUG.match(slug):
            raise ValueError(
                f"invalid project slug {slug!r}: use kebab-case — lowercase letters, "
                "digits and single hyphens (e.g. 'checkout-api')"
            )
        return cls(slug=slug, description=description, created_at=now())
