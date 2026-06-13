"""Composition root: the wired object graph, typed against interfaces."""
from __future__ import annotations

from dataclasses import dataclass

from mnemo.application.ports.embedder import EmbedderPort
from mnemo.application.ports.memory_repository import MemoryRepositoryPort
from mnemo.application.use_cases.interfaces.delete_memory import DeleteMemoryUseCase
from mnemo.application.use_cases.interfaces.remember_memory import RememberMemoryUseCase
from mnemo.application.use_cases.interfaces.search_memory import SearchMemoryUseCase
from mnemo.infrastructure.config import Config


@dataclass
class Container:
    config: Config
    embedder: EmbedderPort
    repository: MemoryRepositoryPort
    remember: RememberMemoryUseCase
    search: SearchMemoryUseCase
    delete: DeleteMemoryUseCase
