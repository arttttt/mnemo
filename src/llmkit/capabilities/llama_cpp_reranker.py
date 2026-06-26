"""Reranker capability over the llama.cpp RANK-mode runtime (GGUF, Metal/CPU).

A cross-encoder reranker scores each (query, document) pair together. For the GGUF runtime the
head is the pre-join: the pair is sent as one ``query<sep>doc`` string (the XLM-R cross-encoder
separator is ``</s></s>``, verified for bge/jina), and the RANK-pooled runtime returns one
relevance logit per pair (higher = more relevant). The model is loaded only for the call,
through the residency manager.
"""
from __future__ import annotations

from collections.abc import Sequence

from llmkit.lifecycle.manager import ResidencyManager
from llmkit.runtime.llama_cpp_rank import LlamaCppRerankRuntime


class LlamaCppReranker:
    def __init__(
        self,
        manager: ResidencyManager[LlamaCppRerankRuntime],
        *,
        sep: str = "</s></s>",
    ) -> None:
        self._manager = manager
        self._sep = sep

    def rank(self, query: str, documents: Sequence[str]) -> list[float]:
        documents = list(documents)
        if not documents:
            return []
        pairs = [f"{query}{self._sep}{document}" for document in documents]
        with self._manager.use() as runtime:
            return runtime.rank(pairs)
