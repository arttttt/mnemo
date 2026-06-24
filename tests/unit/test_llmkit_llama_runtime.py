"""LlamaCppRuntime.unload always ends unloaded — even if the model's close() raises."""
from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

from llmkit.runtime.llama_cpp import GgufSource, LlamaCppRuntime


class _RaisingModel:
    def close(self) -> None:
        raise RuntimeError("native close blew up")


def test_unload_drops_the_handle_even_when_close_raises(caplog):
    runtime = LlamaCppRuntime(GgufSource(model="/nonexistent.gguf"))
    runtime._llama = _RaisingModel()

    with caplog.at_level(logging.WARNING, logger="llmkit.llama"):
        runtime.unload()  # must not raise

    assert runtime._llama is None  # always ends unloaded → a later load() re-initialises
    assert any("close() failed" in record.getMessage() for record in caplog.records)


def test_unload_is_idempotent_when_nothing_is_loaded():
    runtime = LlamaCppRuntime(GgufSource(model="/nonexistent.gguf"))
    runtime.unload()  # no-op, must not raise
    assert runtime._llama is None


def test_unload_logs_current_rss_below_peak_so_a_free_is_visible(caplog, monkeypatch):
    # The bug this guards: a freed model logged the monotonic PEAK, so the line read as if
    # memory had grown after the free. The freed line must report the live current RSS.
    monkeypatch.setattr("llmkit.runtime.llama_cpp.current_rss_mb", lambda: 1800.0)
    monkeypatch.setattr("llmkit.runtime.llama_cpp.peak_rss_mb", lambda: 7300.0)
    runtime = LlamaCppRuntime(GgufSource(model="/x.gguf"))
    runtime._llama = SimpleNamespace(close=lambda: None)

    with caplog.at_level(logging.INFO, logger="llmkit.llama"):
        runtime.unload()

    freed = next(record.getMessage() for record in caplog.records if "freed" in record.getMessage())
    assert "rss=1800MB" in freed and "peak=7300MB" in freed


def test_load_resolves_the_configured_revision(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.gguf").touch()
    captured = {}

    def resolve_snapshot(repo, **kwargs):
        captured.update(repo=repo, **kwargs)
        return str(snapshot)

    class FakeLlama:
        def __init__(self, **kwargs):
            captured["model_path"] = kwargs["model_path"]

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setattr("llmkit.runtime.hf_cache.resolve_snapshot", resolve_snapshot)

    runtime = LlamaCppRuntime(
        GgufSource(
            model="org/model",
            filename="*.gguf",
            revision="immutable-commit",
        )
    )
    runtime.load()

    assert captured["repo"] == "org/model"
    assert captured["revision"] == "immutable-commit"
    assert captured["model_path"] == str(snapshot / "model.gguf")


class _RecordingModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(("chat", kwargs))
        return {"choices": [{"message": {"content": "chat-answer"}}]}

    def create_completion(self, prompt, **kwargs):
        self.calls.append(("raw", {"prompt": prompt, **kwargs}))
        return {"choices": [{"text": "raw-answer"}]}


def test_complete_applies_chat_template_and_sampling_when_chat_enabled():
    runtime = LlamaCppRuntime(
        GgufSource(model="/x.gguf", chat=True, temperature=1.0, top_p=0.95, top_k=64, min_p=0.0)
    )
    runtime._llama = _RecordingModel()

    out = runtime.complete("hello", max_tokens=16)

    assert out == "chat-answer"
    kind, kwargs = runtime._llama.calls[0]
    assert kind == "chat"  # routed through the chat-template path
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert (kwargs["temperature"], kwargs["top_p"], kwargs["top_k"]) == (1.0, 0.95, 64)


def test_complete_uses_raw_completion_when_chat_disabled():
    runtime = LlamaCppRuntime(GgufSource(model="/x.gguf"))  # chat=False (default)
    runtime._llama = _RecordingModel()

    out = runtime.complete("hello", max_tokens=16)

    assert out == "raw-answer"
    assert runtime._llama.calls[0][0] == "raw"  # unchanged raw path for non-chat models
