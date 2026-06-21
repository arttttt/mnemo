"""build_* wires a capability from one config; the source type must match the capability."""
from __future__ import annotations

import pytest

from llmkit.build import build_embedder, build_generator, build_reranker
from llmkit.capabilities.llama_cpp_generator import LlamaCppGenerator
from llmkit.capabilities.onnx_reranker import OnnxReranker
from llmkit.config import ModelConfig
from llmkit.lifecycle.residency import Resident
from llmkit.runtime.llama_cpp import GgufSource
from llmkit.runtime.onnx_encoder import OnnxSource


def test_build_reranker_from_an_onnx_source():
    reranker = build_reranker(ModelConfig(source=OnnxSource(repo="some/repo"), residency=Resident()))
    assert isinstance(reranker, OnnxReranker)  # nothing loaded — construction is lazy


def test_build_embedder_sizes_the_pool():
    embedder = build_embedder(
        ModelConfig(source=OnnxSource(repo="some/repo"), residency=Resident(), pool_size=4),
        dim=8,
    )
    assert embedder._manager._size == 4  # MNEMO_EMBED_WORKERS flows through to the pool size


def test_build_generator_from_a_gguf_source():
    generator = build_generator(ModelConfig(source=GgufSource(model="some/repo")))
    assert isinstance(generator, LlamaCppGenerator)


def test_build_reranker_rejects_a_mismatched_source():
    with pytest.raises(ValueError):
        build_reranker(ModelConfig(source=GgufSource(model="some/repo")))
