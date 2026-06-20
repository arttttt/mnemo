"""OnnxReranker — one score per document, in order; model loaded only for the call."""
from __future__ import annotations

import numpy as np
import pytest

from llmkit.capabilities.onnx_reranker import OnnxReranker
from llmkit.lifecycle.manager import ResidencyManager
from llmkit.lifecycle.residency import Resident, Transient
from llmkit.ports.tokenizer import Encoding
from llmkit.runtime.onnx_encoder import EncoderOutput


class _FakeEncoderRuntime:
    """Stands in for OnnxEncoderRuntime: returns preset logits, tracks load/unload."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def run(self, encodings):
        assert self.loaded  # the manager must have loaded it before the call
        output = np.array(self._scores, dtype=np.float32).reshape(len(encodings), 1)
        return EncoderOutput(output=output, attention_mask=None)


class _FakeTokenizer:
    def count(self, text):
        return len(text.split())

    def encode_batch(self, inputs):
        return [Encoding(ids=[1], attention_mask=[1], type_ids=[0]) for _ in inputs]


def test_reranker_returns_a_score_per_document_in_order():
    runtime = _FakeEncoderRuntime([0.1, 0.9, 0.5])
    reranker = OnnxReranker(ResidencyManager(runtime, Transient()), _FakeTokenizer())

    scores = reranker.rank("q", ["a", "b", "c"])

    assert scores == pytest.approx([0.1, 0.9, 0.5], abs=1e-6)
    assert not runtime.loaded  # transient → freed after the call


def test_reranker_with_no_documents_does_not_load_the_model():
    runtime = _FakeEncoderRuntime([])
    reranker = OnnxReranker(ResidencyManager(runtime, Resident()), _FakeTokenizer())

    assert reranker.rank("q", []) == []
    assert not runtime.loaded  # no work → never loaded
