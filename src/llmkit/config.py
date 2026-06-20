"""One config object describing a model to build: where it is, and how it lives.

The ``source`` type picks the engine (``OnnxSource`` → ONNX encoder, ``GgufSource`` →
llama.cpp); ``residency`` is the load/unload policy, chosen here in code (never from the
environment — a consumer reads its own config and passes it in).
"""
from __future__ import annotations

from dataclasses import dataclass

from llmkit.lifecycle.residency import Residency, Transient
from llmkit.runtime.llama_cpp import GgufSource
from llmkit.runtime.onnx_encoder import OnnxSource

ModelSource = OnnxSource | GgufSource


@dataclass(frozen=True)
class ModelConfig:
    source: ModelSource
    residency: Residency = Transient()
    cache_dir: str | None = None
    pool_size: int = 1   # independent runtime instances kept behind the residency manager
