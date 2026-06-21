"""The embedder pool serves concurrent encodes from independent real sessions (downloads weights)."""
from __future__ import annotations

import concurrent.futures

import pytest

pytestmark = pytest.mark.heavy


def test_pooled_embedder_encodes_concurrently():
    from llmkit.build import build_embedder
    from llmkit.config import ModelConfig
    from llmkit.lifecycle.residency import Resident
    from llmkit.runtime.onnx_encoder import OnnxSource

    source = OnnxSource(repo="Xenova/all-MiniLM-L6-v2", onnx_file="onnx/model.onnx", max_input=256)
    embedder = build_embedder(
        ModelConfig(source=source, residency=Resident(), pool_size=3), dim=384
    )
    texts = [f"sentence number {i}" for i in range(6)]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            vectors = list(pool.map(embedder.encode, texts))
    finally:
        embedder.close()

    assert len(vectors) == 6
    assert all(len(vector) == 384 for vector in vectors)
