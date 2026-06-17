"""The input to a recall run — which project's memory to gather, and how much.

Project-scoped by design (recall answers "what do I know about this project"); the
project's own memories plus globally-scoped ones are included.
"""
from __future__ import annotations

from dataclasses import dataclass

from mnemo.application.pipeline.slot import Slot


@dataclass(frozen=True)
class RecallRequest:
    project: str
    limit: int = 50


RECALL_REQUEST: Slot[RecallRequest] = Slot("recall_request")
