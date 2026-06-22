"""Service shutdown frees the embedder pool when the embedder supports close()."""
from mnemo.adapters.mcp.service import _close_embedder


def test_close_embedder_calls_close_when_present():
    closed = []

    class _Embedder:
        def close(self):
            closed.append(True)

    _close_embedder(_Embedder())

    assert closed == [True]


def test_close_embedder_is_a_noop_when_absent():
    _close_embedder(object())  # the hash embedder has no close() — must not raise
