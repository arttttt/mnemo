import pytest

pytest.importorskip("fastembed")

pytestmark = pytest.mark.heavy


def test_fastembed_is_local_and_deterministic():
    from mnemo.adapters.embedding.fastembed import FastEmbedEmbedder

    embedder = FastEmbedEmbedder()
    first = embedder.encode("hello world")
    second = embedder.encode("hello world")

    assert embedder.dim == len(first) > 0
    assert first == second
