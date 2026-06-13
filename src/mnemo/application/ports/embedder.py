"""Port: turns text into a local embedding vector."""
from typing import Protocol

from mnemo.application.types import Vector


class EmbedderPort(Protocol):
    @property
    def dim(self) -> int: ...

    def encode(self, text: str) -> Vector: ...
