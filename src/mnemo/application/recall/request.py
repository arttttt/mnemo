"""The input to a recall run — which project's memory to recall, the query, and how much.

Recall is always a question: a required, non-empty ``query`` ("where did I leave off on
auth") drives both the retrieval of the relevant memories and the focus of the answer.
``limit`` is the number of most-relevant memories to ground the answer on. Project-scoped
(recall answers "what do I know about this project"); the project's own memories plus
globally-scoped ones are included.
"""
from __future__ import annotations

from dataclasses import dataclass

from mnemo.application.pipeline.slot import Slot
from mnemo.domain.constants import DEFAULT_RECALL_LIMIT


@dataclass(frozen=True)
class RecallRequest:
    project: str
    query: str
    limit: int = DEFAULT_RECALL_LIMIT

    def __post_init__(self) -> None:
        # A blank query would silently degrade recall to an undirected dump, so reject it
        # with an actionable message rather than guessing intent.
        if not self.query or not self.query.strip():
            raise ValueError(
                "recall needs a non-empty query — say what to recall about "
                "(e.g. 'where did I leave off on auth')"
            )


RECALL_REQUEST: Slot[RecallRequest] = Slot("recall_request")
