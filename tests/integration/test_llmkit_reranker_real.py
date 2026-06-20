"""OnnxEncoderRuntime + OnnxReranker against a real ONNX cross-encoder (downloads weights)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.heavy


def test_onnx_reranker_loads_a_real_model_and_scores_each_document():
    from llmkit.capabilities.onnx_reranker import OnnxReranker
    from llmkit.lifecycle.manager import ResidencyManager
    from llmkit.lifecycle.residency import Transient
    from llmkit.runtime.hf_tokenizer import HfTokenizer
    from llmkit.runtime.onnx_encoder import OnnxEncoderRuntime, OnnxSource

    source = OnnxSource(repo="Xenova/ms-marco-MiniLM-L-6-v2", onnx_file="onnx/model.onnx")
    runtime = OnnxEncoderRuntime(source)
    reranker = OnnxReranker(ResidencyManager(runtime, Transient()), HfTokenizer(source))

    scores = reranker.rank("authentication", ["jwt auth tokens", "logging config", "auth session"])

    assert len(scores) == 3
    assert all(isinstance(score, float) for score in scores)
