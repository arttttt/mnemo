"""The eval reranker registry resolves the chosen prod reranker by config, and _rerank_fn reorders
hits by a reranker's scores — both without loading any model (the GGUF build is exercised only by
the heavy domain run)."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from tools.eval.domain import _rerank_fn
from tools.eval.models import RERANKERS, build_reranker


def test_registry_holds_the_prod_reranker():
    assert "bge" in RERANKERS
    assert RERANKERS["bge"].runtime == "gguf"


def test_build_reranker_rejects_an_unknown_name():
    with pytest.raises(SystemExit):
        build_reranker("nope", "")  # fails on the spec lookup, before any runtime import


@dataclass
class _Hit:
    content: str


class _FakeReranker:
    """Scores a doc by how many query words it contains — deterministic, no model."""

    def rank(self, query, docs):
        terms = set(query.lower().split())
        return [float(len(terms & set(d.lower().split()))) for d in docs]


def test_rerank_fn_reorders_hits_by_score():
    rerank = _rerank_fn(_FakeReranker())
    hits = [_Hit("nothing relevant here"), _Hit("the auth jwt rotation path"), _Hit("auth")]
    out = rerank("auth jwt rotation", hits)
    assert out[0].content == "the auth jwt rotation path"  # most query terms -> top
    assert out[-1].content == "nothing relevant here"
