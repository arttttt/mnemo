"""Builds the Container by wiring concrete adapters from config (DI)."""
from __future__ import annotations

from pathlib import Path

from llmkit.ports.generator import Generator
from llmkit.ports.reranker import Reranker

from mnemo.adapters.embedding.sync_embedding_scheduler import SyncEmbeddingScheduler
from mnemo.adapters.session.in_process_session_provider import InProcessSessionProvider
from mnemo.application.ports.embedder import TextEmbedder
from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.ports.project_repository import ProjectRepository
from mnemo.application.ports.session_provider import SessionProvider
from mnemo.application.project_gate import ProjectGate
from mnemo.application.use_cases.browse_memory import BrowseMemoryUseCaseImpl
from mnemo.application.use_cases.create_project import CreateProjectUseCaseImpl
from mnemo.application.use_cases.delete_memory import DeleteMemoryUseCaseImpl
from mnemo.application.use_cases.delete_project import DeleteProjectUseCaseImpl
from mnemo.application.use_cases.list_projects import ListProjectsUseCaseImpl
from mnemo.application.use_cases.recall_project import RecallProjectUseCaseImpl
from mnemo.application.use_cases.remember_memory import RememberMemoryUseCaseImpl
from mnemo.application.use_cases.search_memory import SearchMemoryUseCaseImpl
from mnemo.application.use_cases.update_project import UpdateProjectUseCaseImpl
from mnemo.infrastructure.config import (
    DEFAULT_GENERATOR,
    DEFAULT_GENERATOR_REVISION,
    Config,
)
from mnemo.infrastructure.container import Container
from mnemo.infrastructure.migrations import add_project_foreign_keys, drop_links_table


def build_container(
    config: Config | None = None,
    session_provider: SessionProvider | None = None,
) -> Container:
    config = config or Config.from_env()
    # One-shot, idempotent upgrades, BEFORE the store opens. Disposable — remove each
    # once the live store has it applied (see infrastructure/migrations.py).
    add_project_foreign_keys(config.sqlite_path)
    drop_links_table(config.sqlite_path)
    embedder = _build_embedder(config)
    repository, projects = _build_store(config, embedder.dim)
    session_provider = session_provider or InProcessSessionProvider()
    # Embedding is computed inline by default (CLI / offline). The service swaps in the
    # async scheduler so writes stay cheap (docs/03-architecture.md, deferred embedding).
    scheduler = SyncEmbeddingScheduler(embedder, repository)
    gate = ProjectGate(projects)
    return Container(
        config=config,
        embedder=embedder,
        repository=repository,
        embedding_queue=repository,
        projects=projects,
        scheduler=scheduler,
        remember=RememberMemoryUseCaseImpl(
            repository, scheduler, embedder, session_provider, gate,
        ),
        search=SearchMemoryUseCaseImpl(repository, embedder, gate),
        browse=BrowseMemoryUseCaseImpl(repository, gate),
        recall=RecallProjectUseCaseImpl(
            repository,
            embedder,
            reranker=_build_reranker(config),
            generator=_build_generator(config),
            rerank_top_k=config.rerank_top_k,
            generator_max_tokens=config.generator_max_tokens,
        ),
        delete=DeleteMemoryUseCaseImpl(repository, projects),
        create_project=CreateProjectUseCaseImpl(projects),
        delete_project=DeleteProjectUseCaseImpl(projects),
        update_project=UpdateProjectUseCaseImpl(projects),
        list_projects=ListProjectsUseCaseImpl(projects),
    )


def _build_embedder(config: Config) -> TextEmbedder:
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
            ModelConfig(
                source=source,
                residency=Resident(),
                cache_dir=config.models_dir or None,
                pool_size=config.embed_workers,  # N independent instances → parallel encode
            ),
            dim=1024,
        )
    raise ValueError(f"unknown embedder: {name!r}")


def _build_store(
    config: Config, dim: int
) -> tuple[MemoryRepository, ProjectRepository]:
    """Build the memory store and the project registry together. They SHARE one
    connection (same DB, one writer) so the FK cascade is atomic."""
    from mnemo.adapters.store.sqlite_connections import SqliteConnections
    from mnemo.adapters.store.sqlite_project_repository import (
        SqliteProjectRepositoryImpl,
    )
    from mnemo.adapters.store.sqlite_vec_repository import SqliteRepositoryImpl

    # Build the registry first so `projects` exists before the memories schema that
    # references it; both share the one connection.
    conns = SqliteConnections(config.sqlite_path)
    projects = SqliteProjectRepositoryImpl(conns)
    return SqliteRepositoryImpl(conns, dim), projects


def _build_reranker(config: Config) -> Reranker | None:
    if config.reranker == "off":
        return None
    if config.reranker_revision is None:
        raise ValueError(
            "MNEMO_RERANKER_REVISION is required when MNEMO_RERANKER names a Hugging Face repo"
        )
    from llmkit.build import build_reranker
    from llmkit.config import ModelConfig
    from llmkit.lifecycle.residency import Transient
    from llmkit.runtime.onnx_encoder import OnnxSource

    # cache_dir reuses the models dir so reranker weights live alongside the embedder's.
    return build_reranker(
        ModelConfig(
            source=OnnxSource(repo=config.reranker, revision=config.reranker_revision),
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

    revision = config.generator_revision
    if revision is None and config.generator == DEFAULT_GENERATOR:
        revision = DEFAULT_GENERATOR_REVISION
    if revision is None and not Path(config.generator).exists():
        raise ValueError(
            "MNEMO_GENERATOR_REVISION is required when MNEMO_GENERATOR names a custom "
            "Hugging Face repo"
        )

    return build_generator(
        ModelConfig(
            # Drive the instruct model through its chat template (raw prompts make it ramble)
            # at the vendor-recommended Gemma sampling; a wider context holds the recall bundle.
            source=GgufSource(
                model=config.generator, filename=config.generator_file,
                revision=revision,
                context_tokens=config.generator_context, chat=True,
                temperature=1.0, top_p=0.95, top_k=64, min_p=0.0,
            ),
            residency=Transient(),  # load-on-demand, unload after each recall
            cache_dir=config.models_dir or None,
        )
    )
