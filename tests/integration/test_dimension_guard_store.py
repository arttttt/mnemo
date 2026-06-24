"""verify_store_dimension over a real SQLite store: read the baked CHECK dimension and
fail fast on a genuine mismatch."""
from __future__ import annotations

import pytest

pytest.importorskip("sqlite_vec")

from mnemo.infrastructure.dimension_guard import verify_store_dimension
from tests.support.sqlite_store import open_store


def test_current_dim_reads_the_baked_schema_dimension(tmp_path):
    repo, _ = open_store(tmp_path, dim=8)
    assert repo.current_dim() == 8  # parsed from memories' CHECK(vec_length(embedding) == 8)


def test_guard_passes_on_a_matching_store(tmp_path):
    repo, _ = open_store(tmp_path, dim=8)
    verify_store_dimension(repo, 8)  # no raise


def test_guard_raises_on_a_real_dimension_mismatch(tmp_path):
    repo, _ = open_store(tmp_path, dim=8)
    with pytest.raises(ValueError, match="dimension mismatch"):
        verify_store_dimension(repo, 16)
