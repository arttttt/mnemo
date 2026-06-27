"""Port: count text in tokens — the narrow capability the write-path length guards need.

Split out of ``TokenWindow`` so a caller that only enforces a token cap (e.g. a project
description) depends on the counting alone, not on the embedder's ``max_input`` window (ISP).
"""
from typing import Protocol


class TokenCounter(Protocol):
    def count_tokens(self, text: str) -> int:
        """Untruncated length of ``text`` in tokens — for a write-path cap check."""
        ...
