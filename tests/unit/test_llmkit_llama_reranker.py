"""LlamaCppReranker — pre-joins each pair as ``query<sep>doc``, one score per document in order,
model loaded only for the call."""
from __future__ import annotations

from llmkit.capabilities.llama_cpp_reranker import LlamaCppReranker
from llmkit.lifecycle.manager import ResidencyManager
from llmkit.lifecycle.residency import Resident, Transient


class _FakeRankRuntime:
    """Stands in for LlamaCppRerankRuntime: records the pairs it was handed and returns a preset
    score per pair, tracking load/unload."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.loaded = False
        self.pairs: list[str] | None = None

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def rank(self, pairs):
        assert self.loaded  # the manager must have loaded it before the call
        self.pairs = list(pairs)
        return list(self._scores)


def test_reranker_pre_joins_pairs_and_returns_a_score_per_document_in_order():
    runtime = _FakeRankRuntime([0.1, 0.9, 0.5])
    reranker = LlamaCppReranker(ResidencyManager(lambda: runtime, Transient()))

    scores = reranker.rank("q", ["a", "b", "c"])

    assert scores == [0.1, 0.9, 0.5]  # aligned to the documents, in order
    # the XLM-R cross-encoder separator joins each pair as query</s></s>doc
    assert runtime.pairs == ["q</s></s>a", "q</s></s>b", "q</s></s>c"]
    assert not runtime.loaded  # transient → freed after the call


def test_reranker_honours_a_custom_separator():
    runtime = _FakeRankRuntime([0.0])
    reranker = LlamaCppReranker(ResidencyManager(lambda: runtime, Transient()), sep="[SEP]")

    reranker.rank("q", ["a"])

    assert runtime.pairs == ["q[SEP]a"]


def test_reranker_with_no_documents_does_not_load_the_model():
    runtime = _FakeRankRuntime([])
    reranker = LlamaCppReranker(ResidencyManager(lambda: runtime, Resident()))

    assert reranker.rank("q", []) == []
    assert not runtime.loaded  # no work → never loaded
    assert runtime.pairs is None  # the runtime was never reached
