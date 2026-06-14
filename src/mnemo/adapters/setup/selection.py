"""Parse a user's pick from a numbered list (pure, so it is unit-tested).

Accepts `all`/`*` (everything), `none`/empty (nothing), or a comma/space list of
1-based numbers (e.g. `1,3 4`). Out-of-range and non-numeric tokens are ignored;
the result is the chosen 0-based indices, deduped and ordered.
"""
from __future__ import annotations


def parse_selection(answer: str, count: int) -> list[int]:
    token = answer.strip().lower()
    if token in ("all", "*"):
        return list(range(count))
    if token in ("", "none"):
        return []
    chosen: list[int] = []
    for part in answer.replace(",", " ").split():
        if part.isdigit():
            index = int(part) - 1
            if 0 <= index < count and index not in chosen:
                chosen.append(index)
    return sorted(chosen)
