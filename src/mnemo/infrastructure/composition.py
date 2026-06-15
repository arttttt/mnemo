"""Builds the Container by wiring concrete adapters from config (DI)."""
from __future__ import annotations

from mnemo.adapters.session.in_process_session_provider import InProcessSessionProvider
from mnemo.application.ports.embedder import EmbedderPort
from mnemo.application.ports.memory_repository import MemoryRepositoryPort
from mnemo.application.ports.session_provider import SessionProviderPort
from mnemo.application.use_cases.delete_memory import DeleteMemory
from mnemo.application.use_cases.remember_memory import RememberMemory
from mnemo.application.use_cases.search_memory import SearchMemory
from mnemo.infrastructure.config import Config
from mnemo.infrastructure.container import Container


def build_container(
    config: Config | None = None,
    session_provider: SessionProviderPort | None = None,
) -> Container:
    config = config or Config.from_env()
    embedder = _build_embedder(config.embedder, config.embed_model)
    repository = _build_repository(config)
    session_provider = session_provider or InProcessSessionProvider()
    return Container(
        config=config,
        embedder=embedder,
        repository=repository,
        remember=RememberMemory(repository, embedder, session_provider),
        search=SearchMemory(repository, embedder),
        delete=DeleteMemory(repository),
    )


def _build_embedder(name: str, model: str | None = None) -> EmbedderPort:
    if name == "hash":
        from mnemo.adapters.embedding.hash_embedder import HashEmbedder

        return HashEmbedder()
    if name in ("fastembed", "bge-small", "bge-small-en-v1.5"):
        from mnemo.adapters.embedding.fastembed_embedder import FastEmbedEmbedder

        # MNEMO_EMBED_MODEL picks the concrete fastembed model (e.g. a multilingual one);
        # omit to keep the adapter default. Switching models changes the vector dimension,
        # which is fixed at first write — a different model needs a fresh store.
        return FastEmbedEmbedder(model) if model else FastEmbedEmbedder()
    raise ValueError(f"unknown embedder: {name!r}")


def _build_repository(config: Config) -> MemoryRepositoryPort:
    if config.store == "memory":
        from mnemo.adapters.store.in_memory_repository import InMemoryMemoryRepository

        return InMemoryMemoryRepository(path=config.store_path)
    if config.store == "sqlite":
        from mnemo.adapters.store.sqlite_vec_repository import SqliteVecMemoryRepository

        return SqliteVecMemoryRepository(path=config.sqlite_path)
    raise ValueError(f"unknown store: {config.store!r}")
