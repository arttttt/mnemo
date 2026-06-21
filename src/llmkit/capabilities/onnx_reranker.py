"""Reranker capability over the shared ONNX encoder runtime.

The head is trivial: a relevance cross-encoder emits one logit per (query, document)
pair, higher = more relevant, so the score is that logit. The model is loaded only for
the call, through the residency manager.
"""
from __future__ import annotations

from collections.abc import Sequence

from llmkit.lifecycle.manager import ResidencyManager
from llmkit.ports.tokenizer import Tokenizer
from llmkit.runtime.onnx_encoder import OnnxEncoderRuntime


class OnnxReranker:
    def __init__(
        self, manager: ResidencyManager[OnnxEncoderRuntime], tokenizer: Tokenizer
    ) -> None:
        self._manager = manager
        self._tokenizer = tokenizer

    def rank(self, query: str, documents: Sequence[str]) -> list[float]:
        documents = list(documents)
        if not documents:
            return []
        encodings = self._tokenizer.encode_batch(
            [(query, document) for document in documents]
        )
        with self._manager.use() as runtime:
            logits = runtime.run(encodings).output
        return [float(score) for score in logits.reshape(len(documents), -1)[:, 0]]
