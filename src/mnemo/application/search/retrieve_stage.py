"""Stage: rank memories against the search query — the relevance step, always present.

Reads the seeded ``SEARCH_REQUEST`` and runs the hybrid retrieval — embed the query, fetch
the raw dense + lexical (FTS) legs from the store, then fuse them with the shared ``Fuser``,
filtered by the request's ``criteria``. Produces the fused, scored candidates (``RETRIEVED``)
for presentation AND the per-query confidence ``SIGNALS`` an optional rerank gate reads. When a
reranker is wired (``over_fetch``) the candidate set is widened to a k-scaled pool
(``scaled_pool``) so the reranker has headroom; otherwise it fuses to the request's ``limit``.
The first stage of the retrieve -> (rerank?) -> present pipeline.
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

# Over-fetch pool for reranking: scale with the requested k so the reranker has headroom
# (pool − k) without reading the whole store for a small page. min 20 / cap 50 / factor 5 →
# k=1→20, k=5→25, k=10→50 (the bench/reranker-pool-and-k sweep — NOT a flat constant).
_POOL_MIN, _POOL_CAP, _POOL_FACTOR = 20, 50, 5


def scaled_pool(k: int) -> int:
    return min(_POOL_CAP, max(_POOL_MIN, _POOL_FACTOR * k))


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
        over_fetch: bool = False,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._fuser = fuser
        self._over_fetch = over_fetch

    def run(self, ctx: PipelineContext) -> PipelineContext:
        request = ctx.get(SEARCH_REQUEST)
        n = scaled_pool(request.limit) if self._over_fetch else request.limit
        retrieval = Retrieval(
            criteria=request.criteria,
            limit=n,
            text=request.query,
            vector=self._embedder.encode(request.query),
        )
        channels = self._repository.retrieve_channels(retrieval)
        fused = self._fuser.fuse(channels, n)
        return ctx.set(RETRIEVED, fused.pool).set(SIGNALS, fused.signals)
