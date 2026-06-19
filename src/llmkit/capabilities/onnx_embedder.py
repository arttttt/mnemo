"""Embedder capability over the shared ONNX encoder runtime.

The head: a masked mean-pool of the token embeddings, then optional L2-normalize — a
single sentence vector. (A model that already emits a 2D pooled output is used as-is.)
``dim`` and ``max_input`` come from config; the model loads and frees behind the
residency manager (an embedder is often resident).
"""
from __future__ import annotations

from llmkit.lifecycle.manager import ResidencyManager
from llmkit.runtime.onnx_encoder import EncoderOutput, OnnxEncoderRuntime
from llmkit.types import Vector


class OnnxEmbedder:
    def __init__(
        self,
        manager: ResidencyManager[OnnxEncoderRuntime],
        *,
        dim: int,
        max_input: int,
        normalize: bool = True,
    ) -> None:
        self._manager = manager
        self._dim = dim
        self._max_input = max_input
        self._normalize = normalize

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_input(self) -> int:
        return self._max_input

    def count_tokens(self, text: str) -> int:
        with self._manager.use() as runtime:
            return runtime.count_tokens(text)

    def encode(self, text: str) -> Vector:
        with self._manager.use() as runtime:
            result = runtime.forward([text])
        return self._pool(result)

    def _pool(self, result: EncoderOutput) -> Vector:
        import numpy as np

        output = result.output
        if output.ndim == 3:  # token embeddings [1, seq, dim] → masked mean
            weights = result.attention_mask[:, :, None].astype(np.float32)
            pooled = (output * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        else:  # already a sentence embedding [1, dim]
            pooled = output
        vector = pooled[0]
        if not self._normalize:
            return [float(value) for value in vector]
        norm = float(np.linalg.norm(vector)) or 1.0
        return [float(value) / norm for value in vector]
