"""The input to a search run — the query and the structured filters to rank within.

Search is always a semantic question: a ``query`` drives the hybrid (dense + lexical)
ranking, and ``criteria`` (scope / project / type / tags / related_files / created_after)
filters the candidate set it ranks over. ``limit`` is the page size. The use case owns
turning request params into ``criteria`` and the authorization check; the pipeline owns
retrieval and presentation — so the criteria arrive here already built.
"""
from __future__ import annotations

from dataclasses import dataclass

from mnemo.application.pipeline.slot import Slot
from mnemo.application.search_criteria import SearchCriteria


@dataclass(frozen=True)
class SearchRequest:
    criteria: SearchCriteria
    query: str
    limit: int = 10


SEARCH_REQUEST: Slot[SearchRequest] = Slot("search_request")
