"""OnnxEmbedder against a real ONNX sentence encoder (downloads weights)."""
from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.heavy


def test_embedder_encodes_a_real_model_to_a_unit_vector():
    from llmkit.capabilities.onnx_embedder import OnnxEmbedder
    from llmkit.lifecycle.manager import ResidencyManager
    from llmkit.lifecycle.residency import Resident
    from llmkit.runtime.hf_tokenizer import HfTokenizer
    from llmkit.runtime.onnx_encoder import OnnxEncoderRuntime, OnnxSource

    source = OnnxSource(repo="Xenova/all-MiniLM-L6-v2", onnx_file="onnx/model.onnx", max_input=256)
    embedder = OnnxEmbedder(
        ResidencyManager(OnnxEncoderRuntime(source), Resident()), HfTokenizer(source),
        dim=384, max_input=256,
    )

    vector = embedder.encode("hello world")

    assert len(vector) == 384
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, abs_tol=1e-3)  # L2-normalized
