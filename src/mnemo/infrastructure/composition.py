"""Builds the Container by wiring concrete adapters from config (DI)."""
from __future__ import annotations

from llmkit.ports.generator import Generator
from llmkit.ports.reranker import Reranker

from mnemo.adapters.embedding.sync_embedding_scheduler import SyncEmbeddingScheduler
from mnemo.adapters.session.in_process_session_provider import InProcessSessionProvider
from mnemo.application.ports.embedder import EmbedderPort
from mnemo.application.ports.memory_repository import MemoryRepositoryPort
from mnemo.application.ports.session_provider import SessionProviderPort
from mnemo.application.use_cases.browse_memory import BrowseMemory
from mnemo.application.use_cases.delete_memory import DeleteMemory
from mnemo.application.use_cases.recall_project import RecallProject
from mnemo.application.use_cases.remember_memory import RememberMemory
from mnemo.application.use_cases.search_memory import SearchMemory
from mnemo.infrastructure.config import Config
from mnemo.infrastructure.container import Container


def build_container(
    config: Config | None = None,
    session_provider: SessionProviderPort | None = None,
) -> Container:
    config = config or Config.from_env()
    embedder = _build_embedder(config)
    repository = _build_repository(config, embedder.dim)
    session_provider = session_provider or InProcessSessionProvider()
    # Embedding is computed inline by default (CLI / offline). The service swaps in the
    # async scheduler so writes stay cheap (docs/03-architecture.md, deferred embedding).
    scheduler = SyncEmbeddingScheduler(embedder, repository)
    return Container(
        config=config,
        embedder=embedder,
        repository=repository,
        scheduler=scheduler,
        remember=RememberMemory(repository, scheduler, session_provider),
        search=SearchMemory(repository, embedder),
        browse=BrowseMemory(repository),
        recall=RecallProject(
            repository,
            reranker=_build_reranker(config),
            generator=_build_generator(config),
            rerank_top_k=config.rerank_top_k,
            generator_max_tokens=config.generator_max_tokens,
        ),
        delete=DeleteMemory(repository),
    )


def _build_embedder(config: Config) -> EmbedderPort:
    name = config.embedder
    if name == "hash":
        from mnemo.adapters.embedding.hash_embedder import HashEmbedder

        return HashEmbedder()
    if name == "pplx":  # the default — pplx-embed-v1-0.6b int8 ONNX (CPU), via llmkit
        from llmkit.build import build_embedder
        from llmkit.config import ModelConfig
        from llmkit.lifecycle.residency import Resident
        from llmkit.runtime.onnx_encoder import OnnxSource

        source = OnnxSource(
            repo="perplexity-ai/pplx-embed-v1-0.6b",
            onnx_file="onnx/model_quantized.onnx",   # int8
            revision="2c4d510dd4a732063c31a0f70193e35067b51fd8",  # pinned: switching = a reindex
            max_input=config.embed_max_tokens,
        )
        return build_embedder(
            ModelConfig(source=source, residency=Resident(), cache_dir=config.models_dir or None),
            dim=1024,
        )
    if name in ("fastembed", "bge-small", "bge-small-en-v1.5"):
        from mnemo.adapters.embedding.fastembed_embedder import FastEmbedEmbedder

        # MNEMO_EMBED_MODEL picks the concrete fastembed model (e.g. a multilingual one);
        # omit to keep the adapter default. Switching models changes the vector dimension,
        # which is fixed at first write — a different model needs a reindex.
        return FastEmbedEmbedder(config.embed_model) if config.embed_model else FastEmbedEmbedder()
    raise ValueError(f"unknown embedder: {name!r}")


def _build_repository(config: Config, dim: int) -> MemoryRepositoryPort:
    if config.store == "memory":
        from mnemo.adapters.store.in_memory_repository import InMemoryMemoryRepository

        return InMemoryMemoryRepository(path=config.store_path)
    if config.store == "sqlite":
        from mnemo.adapters.store.sqlite_vec_repository import SqliteVecMemoryRepository

        # dim up front lets a pending (vector-less) first write create the schema.
        return SqliteVecMemoryRepository(path=config.sqlite_path, dim=dim)
    raise ValueError(f"unknown store: {config.store!r}")


def _build_reranker(config: Config) -> Reranker | None:
    if config.reranker == "off":
        return None
    from llmkit.build import build_reranker
    from llmkit.config import ModelConfig
    from llmkit.lifecycle.residency import Transient
    from llmkit.runtime.onnx_encoder import OnnxSource

    # cache_dir reuses the models dir so reranker weights live alongside the embedder's.
    return build_reranker(
        ModelConfig(
            source=OnnxSource(repo=config.reranker),
            residency=Transient(),
            cache_dir=config.models_dir or None,
        )
    )


def _build_generator(config: Config) -> Generator | None:
    if config.generator == "off":
        return None
    from llmkit.build import build_generator
    from llmkit.config import ModelConfig
    from llmkit.lifecycle.residency import Transient
    from llmkit.runtime.llama_cpp import GgufSource

    return build_generator(
        ModelConfig(
            source=GgufSource(model=config.generator, filename=config.generator_file),
            residency=Transient(),
            cache_dir=config.models_dir or None,
        )
    )
