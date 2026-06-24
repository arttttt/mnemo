"""verify_store_dimension: fail fast when the embedder and store dimensions disagree."""
from __future__ import annotations

import pytest

from mnemo.infrastructure.dimension_guard import verify_store_dimension


class _Queue:
    """A minimal EmbeddingQueue stand-in exposing only the dimension the guard reads."""

    def __init__(self, dim):
        self._dim = dim

    def current_dim(self):
        return self._dim


def test_passes_when_dimensions_match():
    verify_store_dimension(_Queue(1024), 1024)  # no raise


def test_passes_on_a_fresh_store_with_no_baked_dimension():
    verify_store_dimension(_Queue(None), 1024)  # None → nothing baked yet, not a mismatch


def test_raises_with_an_actionable_message_on_mismatch():
    with pytest.raises(ValueError) as excinfo:
        verify_store_dimension(_Queue(1024), 384)
    message = str(excinfo.value)
    assert "1024" in message and "384" in message  # both dimensions named
    assert "reindex" in message                     # and the fix is spelled out
