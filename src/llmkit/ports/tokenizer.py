"""Tokenizer port: turn text (or a text pair) into token ids — the cheap step, kept
separate from the heavy session. Injected into the ONNX encoder capabilities so it is
swappable (a different tokenizer, a test fake) without touching the capability, and so a
token count never has to lease a model instance from the session pool.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

TokenizerInput = str | tuple[str, str]


@dataclass(frozen=True)
class Encoding:
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]


class Tokenizer(Protocol):
    def count(self, text: str) -> int:
        """Untruncated token count for one text (the write-path over-window check)."""
        ...

    def encode_batch(self, inputs: Sequence[TokenizerInput]) -> list[Encoding]:
        """Tokenize a batch of single texts, or (left, right) pairs for a cross-encoder."""
        ...
