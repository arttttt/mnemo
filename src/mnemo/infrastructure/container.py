"""Composition root: the wired object graph, typed against interfaces."""
from __future__ import annotations

from dataclasses import dataclass

from mnemo.application.ports.embedder import TextEmbedder
from mnemo.application.ports.embedding_queue import EmbeddingQueue
from mnemo.application.ports.embedding_scheduler import EmbeddingScheduler
from mnemo.application.ports.memory_repository import MemoryRepository
from mnemo.application.ports.project_repository import ProjectRepository
from mnemo.application.use_cases.interfaces.browse_memory import BrowseMemoryUseCase
from mnemo.application.use_cases.interfaces.delete_memory import DeleteMemoryUseCase
from mnemo.application.use_cases.interfaces.recall_project import RecallProjectUseCase
from mnemo.application.use_cases.interfaces.remember_memory import RememberMemoryUseCase
from mnemo.application.use_cases.interfaces.search_memory import SearchMemoryUseCase
from mnemo.infrastructure.config import Config


@dataclass
class Container:
    config: Config
    embedder: TextEmbedder
    repository: MemoryRepository
    embedding_queue: EmbeddingQueue  # same store object, the deferred-embedding facet
    projects: ProjectRepository
    scheduler: EmbeddingScheduler
    remember: RememberMemoryUseCase
    search: SearchMemoryUseCase
    browse: BrowseMemoryUseCase
    recall: RecallProjectUseCase
    delete: DeleteMemoryUseCase
