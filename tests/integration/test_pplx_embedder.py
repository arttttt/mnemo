"""Real PplxEmbedder — fetches the int8 model and runs it (heavy + networked)."""
import pytest

pytestmark = pytest.mark.heavy


def test_pplx_embedder_encodes_and_ranks(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    pytest.importorskip("huggingface_hub")
    from mnemo.adapters.embedding.pplx_embedder import PplxEmbedder

    embedder = PplxEmbedder(models_dir=str(tmp_path))
    assert embedder.dim == 1024
    assert embedder.max_input == 2048

    query = embedder.encode("how do we handle auth errors")
    auth = embedder.encode("JWT refresh token rotation with httpOnly cookies")
    postgres = embedder.encode("postgres connection pool tuning")

    def cosine(a, b):  # vectors are already L2-normalized
        return sum(x * y for x, y in zip(a, b))

    assert len(query) == 1024
    assert cosine(query, auth) > cosine(query, postgres)  # the auth doc ranks higher
    # the full (untruncated) token count drives the over-window reject
    assert embedder.count_tokens("one two three four five") >= 5
