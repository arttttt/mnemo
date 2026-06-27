"""Build a capability from one config: source → runtime, residency → manager, kind → head.

``build_*`` are the typed public entry points. The encoder capabilities share the same
internal wiring (``_onnx_manager``), so the ONNX runtime is set up the same way for every
one of them; the generator has its own runtime.
"""
from __future__ import annotations

from llmkit.capabilities.llama_cpp_generator import LlamaCppGenerator
from llmkit.capabilities.llama_cpp_reranker import LlamaCppReranker
from llmkit.capabilities.onnx_embedder import OnnxEmbedder
from llmkit.capabilities.onnx_reranker import OnnxReranker
from llmkit.config import ModelConfig
from llmkit.lifecycle.manager import ResidencyManager
from llmkit.ports.embedder import Embedder
from llmkit.ports.generator import Generator
from llmkit.ports.reranker import Reranker
from llmkit.runtime.hf_tokenizer import HfTokenizer
from llmkit.runtime.llama_cpp import GgufSource, LlamaCppRuntime
from llmkit.runtime.llama_cpp_rank import LlamaCppRerankRuntime
from llmkit.runtime.onnx_encoder import OnnxEncoderRuntime, OnnxSource


def _onnx_manager(config: ModelConfig) -> ResidencyManager[OnnxEncoderRuntime]:
    if not isinstance(config.source, OnnxSource):
        raise ValueError("an ONNX-encoder capability needs an OnnxSource")
    source, cache = config.source, config.cache_dir
    return ResidencyManager(
        lambda: OnnxEncoderRuntime(source, cache_dir=cache),
        config.residency,
        size=config.pool_size,
    )


def _onnx_tokenizer(config: ModelConfig) -> HfTokenizer:
    if not isinstance(config.source, OnnxSource):
        raise ValueError("an ONNX-encoder capability needs an OnnxSource")
    return HfTokenizer(config.source, cache_dir=config.cache_dir)


def build_embedder(config: ModelConfig, *, dim: int, normalize: bool = True) -> Embedder:
    source = config.source
    if not isinstance(source, OnnxSource):
        raise ValueError("an embedder needs an OnnxSource")
    return OnnxEmbedder(
        _onnx_manager(config), _onnx_tokenizer(config),
        dim=dim, max_input=source.max_input, normalize=normalize,
    )


def build_reranker(config: ModelConfig) -> Reranker:
    # The source type picks the cross-encoder backend: a GGUF runs on llama.cpp in RANK mode
    # (GPU/Metal), an ONNX source runs on the shared CPU encoder. The GGUF runtime pre-joins
    # the pair itself, so it needs no tokenizer.
    if isinstance(config.source, GgufSource):
        source, cache = config.source, config.cache_dir
        return LlamaCppReranker(
            ResidencyManager(
                lambda: LlamaCppRerankRuntime(source, cache_dir=cache),
                config.residency,
                size=config.pool_size,
            )
        )
    return OnnxReranker(_onnx_manager(config), _onnx_tokenizer(config))


def build_generator(config: ModelConfig) -> Generator:
    if not isinstance(config.source, GgufSource):
        raise ValueError("a generator needs a GgufSource")
    source, cache = config.source, config.cache_dir
    return LlamaCppGenerator(
        ResidencyManager(
            lambda: LlamaCppRuntime(source, cache_dir=cache),
            config.residency,
            size=config.pool_size,
        )
    )
