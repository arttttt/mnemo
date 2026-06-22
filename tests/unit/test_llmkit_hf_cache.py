"""Cache-first model resolution — use the local cache, reach the hub only on a miss."""
from __future__ import annotations

import pytest

huggingface_hub = pytest.importorskip("huggingface_hub")
from huggingface_hub.utils import LocalEntryNotFoundError

from llmkit.runtime.hf_cache import resolve_snapshot


def test_resolves_from_cache_without_downloading_when_present(monkeypatch):
    calls = []

    def fake_snapshot_download(repo, **kwargs):
        calls.append(kwargs)
        return "/cache/snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    path = resolve_snapshot("repo/x", cache_dir="/c", allow_patterns=["*.gguf"])

    assert path == "/cache/snapshot"
    assert len(calls) == 1  # a single cache lookup, no download
    assert calls[0]["local_files_only"] is True  # the hub is never reached


def test_downloads_once_when_not_cached(monkeypatch):
    calls = []

    def fake_snapshot_download(repo, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("not cached")
        return "/downloaded/snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    path = resolve_snapshot("repo/x", cache_dir="/c")

    assert path == "/downloaded/snapshot"
    assert len(calls) == 2  # cache miss → exactly one real download
    assert calls[0]["local_files_only"] is True
    assert "local_files_only" not in calls[1]  # the fallback download is not local-only
