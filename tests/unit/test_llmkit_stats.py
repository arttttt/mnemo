"""Process-memory helpers: a live (droppable) current RSS distinct from the monotonic peak."""
from __future__ import annotations

from llmkit.runtime import _stats


def test_current_rss_mb_is_a_positive_plausible_reading():
    rss = _stats.current_rss_mb()
    assert 0 < rss < 1_000_000  # resident, and not a unit-blunder (e.g. pages read as MiB)


def test_current_rss_mb_falls_back_to_peak_when_the_reader_fails(monkeypatch):
    # A logging helper must never raise: force the macOS/BSD branch, blow up `ps`, and the
    # helper must degrade to the peak rather than propagate.
    monkeypatch.setattr(_stats.sys, "platform", "darwin")

    def boom(*args, **kwargs):
        raise OSError("ps unavailable")

    monkeypatch.setattr(_stats.subprocess, "run", boom)
    assert _stats.current_rss_mb() == _stats.peak_rss_mb()
