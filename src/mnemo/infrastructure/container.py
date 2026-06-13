"""Composition root: wire concrete adapters into use cases from config (DI)."""
from __future__ import annotations

from dataclasses import dataclass

from mnemo.application.ports import EmbedderPort, MemoryRepositoryPort
from mnemo.application.use_cases import RememberMemory, SearchMemory
from mnemo.infrastructure.config import Config


@dataclass
class Container:
    config: Config
    embedder: EmbedderPort
    repository: MemoryRepositoryPort
    remember: RememberMemory
    search: SearchMemory


def build_container(config: Config | None = None) -> Container:
    config = config or Config.from_env()
    embedder = _build_embedder(config.embedder)
    repository = _build_repository(config)
    return Container(
        config=config,
        embedder=embedder,
        repository=repository,
        remember=RememberMemory(repository, embedder),
        search=SearchMemory(repository, embedder),
    )


def _build_embedder(name: str) -> EmbedderPort:
    if name == "hash":
        from mnemo.adapters.embedding.hashing import HashEmbedder

        return HashEmbedder()
    if name in ("fastembed", "bge-small", "bge-small-en-v1.5"):
        from mnemo.adapters.embedding.fastembed import FastEmbedEmbedder

        return FastEmbedEmbedder()
    raise ValueError(f"unknown embedder: {name!r}")


def _build_repository(config: Config) -> MemoryRepositoryPort:
    if config.store == "memory":
        from mnemo.adapters.store.in_memory import InMemoryMemoryRepository

        return InMemoryMemoryRepository(path=config.store_path)
    if config.store == "lancedb":
        raise NotImplementedError("LanceMemoryRepository arrives in Phase 1")
    raise ValueError(f"unknown store: {config.store!r}")
