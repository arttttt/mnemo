"""Stage: rank memories against the search query — the relevance step, always present.

Reads the seeded ``SEARCH_REQUEST`` and runs the hybrid retrieval — embed the query, fetch
the raw dense + lexical (FTS) legs from the store, then fuse them with the shared ``Fuser``,
filtered by the request's ``criteria``. Produces the fused, scored candidates (``RETRIEVED``)
for presentation AND the per-query confidence ``SIGNALS`` an optional rerank gate reads. A
``pool`` over-fetch (set when a reranker is wired) widens the candidate set the fuser ranks
over; otherwise it fuses to the request's ``limit``. The first stage of the
retrieve -> (rerank?) -> present pipeline.
"""
from __future__ import annotations

from mnemo.application.fusion.fuser import Fuser
from mnemo.application.fusion.results import RetrievalSignals
from mnemo.application.pipeline.context import PipelineContext
from mnemo.application.pipeline.slot import Slot
from mnemo.application.ports.embedder import TextEmbedder
from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.retrieval import Retrieval
from mnemo.application.scored_memory import ScoredMemory
from mnemo.application.search.request import SEARCH_REQUEST

RETRIEVED: Slot[tuple[ScoredMemory, ...]] = Slot("retrieved")
# The confidence signals live with their producer (this stage); the rerank gate reads them.
SIGNALS: Slot[RetrievalSignals] = Slot("signals")


class RetrieveStage:
    key = "retrieve"
    requires = frozenset({SEARCH_REQUEST.name})
    provides = frozenset({RETRIEVED.name, SIGNALS.name})

    def __init__(
        self,
        repository: MemoryRepository,
        embedder: TextEmbedder,
        fuser: Fuser,
        *,
        pool: int | None = None,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._fuser = fuser
        self._pool = pool

    def run(self, ctx: PipelineContext) -> PipelineContext:
        request = ctx.get(SEARCH_REQUEST)
        n = self._pool or request.limit
        retrieval = Retrieval(
            criteria=request.criteria,
            limit=n,
            text=request.query,
            vector=self._embedder.encode(request.query),
        )
        channels = self._repository.retrieve_channels(retrieval)
        fused = self._fuser.fuse(channels, n)
        return ctx.set(RETRIEVED, fused.pool).set(SIGNALS, fused.signals)
