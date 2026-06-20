"""Port: turns text into a local embedding vector."""
from typing import Protocol

from mnemo.application.ports.token_window import TokenWindow
from mnemo.application.types import Vector


class TextEmbedder(TokenWindow, Protocol):
    @property
    def dim(self) -> int: ...

    def encode(self, text: str) -> Vector: ...
