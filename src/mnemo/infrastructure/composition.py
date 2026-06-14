"""Builds the Container by wiring concrete adapters from config (DI)."""
from __future__ import annotations

from mnemo.adapters.session.in_process_session_provider import InProcessSessionProvider
from mnemo.application.ports.embedder import EmbedderPort
from mnemo.application.ports.memory_repository import MemoryRepositoryPort
from mnemo.application.use_cases.delete_memory import DeleteMemory
from mnemo.application.use_cases.interfaces.migrate_memories import MigrateMemoriesUseCase
from mnemo.application.use_cases.migrate_memories import MigrateMemories
from mnemo.application.use_cases.remember_memory import RememberMemory
from mnemo.application.use_cases.search_memory import SearchMemory
from mnemo.infrastructure.config import Config
from mnemo.infrastructure.container import Container


def build_container(config: Config | None = None) -> Container:
    config = config or Config.from_env()
    embedder = _build_embedder(config.embedder)
    repository = _build_repository(config)
    session_provider = InProcessSessionProvider()
    return Container(
        config=config,
        embedder=embedder,
        repository=repository,
        remember=RememberMemory(repository, embedder, session_provider),
        search=SearchMemory(repository, embedder),
        delete=DeleteMemory(repository),
    )


def build_migration(config: Config | None = None) -> MigrateMemoriesUseCase:
    """Wire the one-off migration from the JSON store into the LanceDB store."""
    config = config or Config.from_env()
    from mnemo.adapters.store.in_memory_repository import InMemoryMemoryRepository
    from mnemo.adapters.store.lancedb_repository import LanceDbMemoryRepository

    source = InMemoryMemoryRepository(path=config.store_path)
    target = LanceDbMemoryRepository(uri=config.lancedb_uri)
    return MigrateMemories(source, target, _build_embedder(config.embedder))


def _build_embedder(name: str) -> EmbedderPort:
    if name == "hash":
        from mnemo.adapters.embedding.hash_embedder import HashEmbedder

        return HashEmbedder()
    if name in ("fastembed", "bge-small", "bge-small-en-v1.5"):
        from mnemo.adapters.embedding.fastembed_embedder import FastEmbedEmbedder

        return FastEmbedEmbedder()
    raise ValueError(f"unknown embedder: {name!r}")


def _build_repository(config: Config) -> MemoryRepositoryPort:
    if config.store == "memory":
        from mnemo.adapters.store.in_memory_repository import InMemoryMemoryRepository

        return InMemoryMemoryRepository(path=config.store_path)
    if config.store == "lancedb":
        from mnemo.adapters.store.lancedb_repository import LanceDbMemoryRepository

        return LanceDbMemoryRepository(uri=config.lancedb_uri)
    raise ValueError(f"unknown store: {config.store!r}")
