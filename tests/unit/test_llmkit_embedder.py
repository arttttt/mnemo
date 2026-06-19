"""OnnxEmbedder — masked mean-pool + normalize; dim/max_input from config; loads behind manager."""
from __future__ import annotations

import numpy as np
import pytest

from llmkit.capabilities.onnx_embedder import OnnxEmbedder
from llmkit.lifecycle.manager import ResidencyManager
from llmkit.lifecycle.residency import Resident
from llmkit.runtime.onnx_encoder import EncoderOutput


class _FakeEncoderRuntime:
    def __init__(self, output=None, mask=None) -> None:
        self._output = output
        self._mask = mask
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def forward(self, texts):
        assert self.loaded  # the manager must have loaded it before the call
        return EncoderOutput(output=self._output, attention_mask=self._mask)

    def count_tokens(self, text):
        assert self.loaded
        return len(text.split())


def _embedder(runtime) -> OnnxEmbedder:
    return OnnxEmbedder(ResidencyManager(runtime, Resident()), dim=3, max_input=512)


def test_embedder_mean_pools_over_the_mask_then_normalizes():
    # 1 text, seq=2, hidden=3; both tokens unmasked → mean [2,0,0] → normalized [1,0,0]
    output = np.array([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], dtype=np.float32)
    mask = np.array([[1, 1]], dtype=np.int64)
    embedder = _embedder(_FakeEncoderRuntime(output, mask))

    assert embedder.encode("x") == pytest.approx([1.0, 0.0, 0.0])
    assert embedder.dim == 3 and embedder.max_input == 512


def test_pooling_ignores_masked_padding_tokens():
    # the second token is padding (mask 0) → mean is just the first token, then normalized
    output = np.array([[[3.0, 4.0, 0.0], [99.0, 99.0, 99.0]]], dtype=np.float32)
    mask = np.array([[1, 0]], dtype=np.int64)
    embedder = _embedder(_FakeEncoderRuntime(output, mask))

    assert embedder.encode("x") == pytest.approx([0.6, 0.8, 0.0])  # [3,4,0] / 5


def test_count_tokens_delegates_to_the_runtime():
    embedder = _embedder(_FakeEncoderRuntime())
    assert embedder.count_tokens("a b c") == 3
