"""Cache-first Hugging Face snapshot resolution — load from the local cache, fetch only on a miss.

llmkit is offline-first: once a model's files are cached, loading must not touch the hub — no
network round-trip, no "Fetching" progress noise. ``snapshot_download``/``from_pretrained`` do
not do this on their own (they always run the resolution path and print bars even on a cache
hit, and an unpinned repo is re-resolved against the hub every load). This resolves a repo from
the local cache first and downloads — once, with a one-line notice and the real progress bar —
only when the files are absent.
"""
from __future__ import annotations

import logging

_log = logging.getLogger("llmkit.hf")


def resolve_snapshot(
    repo: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    allow_patterns: list[str] | None = None,
) -> str:
    """Return the local snapshot directory for ``repo``, downloading only if it is not cached."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import (
        LocalEntryNotFoundError,
        disable_progress_bars,
        enable_progress_bars,
    )

    kwargs = {"revision": revision, "cache_dir": cache_dir, "allow_patterns": allow_patterns}
    disable_progress_bars()  # a cache hit must be silent — no bar, no hub round-trip
    try:
        return snapshot_download(repo, local_files_only=True, **kwargs)
    except LocalEntryNotFoundError:
        _log.info("downloading %s (first run, this may take a while)…", repo)
        enable_progress_bars()  # show progress for the real, one-time download
        return snapshot_download(repo, **kwargs)
    finally:
        enable_progress_bars()
