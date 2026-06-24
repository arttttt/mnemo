"""Startup guard: the configured embedder must match the store's baked dimension.

The store bakes its embedding dimension at first write (``CHECK(vec_length(embedding) == N)``);
switching ``MNEMO_EMBEDDER`` to a different-dimension model **without** running ``mnemo reindex``
otherwise fails deep and opaquely — a ``CHECK`` violation on the next write, or sqlite-vec's
"Vector dimension mismatch" on the next query. This turns that into a fail-fast, actionable error.

Run it at **service start** and on the read/write CLI commands — **never** inside
``build_container`` or the ``reindex`` path, which must open a mismatched store precisely to
repair it via ``set_dimension``.
"""
from __future__ import annotations

from mnemo.application.ports.embedding_queue import EmbeddingQueue


def verify_store_dimension(queue: EmbeddingQueue, embedder_dim: int) -> None:
    """Raise ``ValueError`` when the store's baked dimension disagrees with the embedder.

    No-op on a fresh store (nothing baked yet) or when the two already match — only a genuine
    mismatch raises, with the fix spelled out. mnemo never auto-reindexes (it rewrites every
    vector), so the resolution is the caller's to choose.
    """
    stored = queue.current_dim()
    if stored is not None and stored != embedder_dim:
        raise ValueError(
            f"embedder/store dimension mismatch: the store is dim {stored}, the configured "
            f"embedder is dim {embedder_dim}. Run `mnemo reindex` to migrate the store to "
            f"{embedder_dim}, or pin MNEMO_EMBEDDER to a {stored}-dim model."
        )
