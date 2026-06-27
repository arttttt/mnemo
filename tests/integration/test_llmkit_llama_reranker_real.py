"""LlamaCppRerankRuntime + LlamaCppReranker against the real bge GGUF cross-encoder.

Needs llama-cpp and downloads the Q8 GGUF; gated by the heavy marker. Asserts the cross-encoder
contract: a relevant document outscores irrelevant ones, non-degenerately (the same check
``tools/eval/rerankers.sanity_check`` runs before an A/B — it catches a wrong separator or a
broken pair join).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.heavy


def test_llama_reranker_loads_the_real_bge_gguf_and_ranks_a_relevant_doc_first(tmp_path):
    from llmkit.capabilities.llama_cpp_reranker import LlamaCppReranker
    from llmkit.lifecycle.manager import ResidencyManager
    from llmkit.lifecycle.residency import Transient
    from llmkit.runtime.llama_cpp import GgufSource
    from llmkit.runtime.llama_cpp_rank import LlamaCppRerankRuntime

    source = GgufSource(
        model="gpustack/bge-reranker-v2-m3-GGUF",
        filename="*Q8_0.gguf",
        revision="3093af03b1a635e67b084b1d8c03c5f5e020fd05",
        context_tokens=1024,
    )
    # An isolated cache_dir so the test is hermetic — it downloads the Q8 GGUF fresh rather than
    # leaning on whatever is in the user's default HF cache (which may hold a partial snapshot).
    reranker = LlamaCppReranker(
        ResidencyManager(
            lambda: LlamaCppRerankRuntime(source, cache_dir=str(tmp_path)), Transient()
        )
    )

    query = "When did Caroline go to the LGBTQ support group?"
    docs = [
        "Caroline (3 May 2023): I went to the LGBTQ support group on 7 May 2023.",
        "Melanie: I painted a sunrise this morning, the colors were beautiful.",
        "Caroline: The weather is really nice today, going for a run.",
    ]

    scores = reranker.rank(query, docs)

    assert len(scores) == 3
    assert all(isinstance(score, float) for score in scores)
    assert scores[0] == max(scores)  # the relevant doc ranks first
    assert max(abs(s) for s in scores) > 1e-3  # non-degenerate (not a broken sep/join)
