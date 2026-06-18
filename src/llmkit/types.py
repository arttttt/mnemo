"""Shared value types for llmkit's capability ports."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Vector = Sequence[float]


@dataclass(frozen=True)
class NliScores:
    """Probabilities of the three NLI relations for a (premise, hypothesis) pair."""

    entailment: float
    neutral: float
    contradiction: float
