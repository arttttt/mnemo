"""Shared TIER-1 near-match — difflib over a candidate list, on error paths only.

So a typo'd handle suggests the real one instead of failing silently. Used by the project
gate (over registered slugs) and by `get` (over topic_keys). TIER-2 (semantic, over
descriptions) is deferred — see the project-entity/near-match-tier2 note.
"""
from __future__ import annotations

import difflib

_MAX_CANDIDATES = 5


def near_matches(needle: str, candidates: list[str]) -> list[str]:
    """The candidates closest to `needle` — difflib top-N, no threshold (recovery beats
    hiding suggestions)."""
    return difflib.get_close_matches(needle, candidates, n=_MAX_CANDIDATES, cutoff=0.0)


def did_you_mean(candidates: list[str]) -> str:
    """A ' Did you mean: a, b?' suffix for a not-found error, or '' when there are none."""
    return f" Did you mean: {', '.join(candidates)}?" if candidates else ""
