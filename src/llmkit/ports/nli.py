"""Port: natural-language inference over a text pair (entailment / neutral / contradiction)."""
from __future__ import annotations

from typing import Protocol

from llmkit.types import NliScores


class Nli(Protocol):
    def classify(self, premise: str, hypothesis: str) -> NliScores: ...
