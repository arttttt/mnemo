"""OnnxEncoderRuntime.unload reports the live RSS, so a free is visible (not the monotonic peak)."""
from __future__ import annotations

import logging

from llmkit.runtime.onnx_encoder import OnnxEncoderRuntime, OnnxSource


def test_unload_logs_current_rss_below_peak_so_a_free_is_visible(caplog, monkeypatch):
    # The bug this guards: a freed model logged the monotonic PEAK, so the line read as if the
    # encoder had grown. The freed line must report the live (droppable) current RSS.
    monkeypatch.setattr("llmkit.runtime.onnx_encoder.current_rss_mb", lambda: 700.0)
    monkeypatch.setattr("llmkit.runtime.onnx_encoder.peak_rss_mb", lambda: 1500.0)
    runtime = OnnxEncoderRuntime(OnnxSource(repo="org/encoder"))
    runtime._session = object()  # a loaded session so unload proceeds to the freed log

    with caplog.at_level(logging.INFO, logger="llmkit.onnx"):
        runtime.unload()

    freed = next(record.getMessage() for record in caplog.records if "freed" in record.getMessage())
    assert "rss=700MB" in freed and "peak=1500MB" in freed
    assert runtime._session is None  # always ends unloaded
