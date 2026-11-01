"""Shared test setup."""
import logging
import os
import threading

import pytest


@pytest.fixture(autouse=True)
def _default_model_stages_off(monkeypatch):
    """Keep the offline suite model-free by DEFAULT. The product now defaults the reranker (and
    already defaulted the generator) to a real model, so a test that builds a container from the
    environment (``Config.from_env``) without an explicit override would otherwise try to
    download/load a GGUF. Default both stages to "off" here; a test that wants a real model sets
    the env var itself (e.g. the heavy tests, which build sources directly anyway) and overrides
    this. Tests that construct ``Config(...)`` directly bypass the environment, so they set
    ``reranker="off"`` at the call site instead."""
    for var in ("MNEMO_RERANKER", "MNEMO_GENERATOR"):
        if var not in os.environ:
            monkeypatch.setenv(var, "off")
    yield


@pytest.fixture(autouse=True)
def _client_configs_stay_under_the_test_home(monkeypatch):
    """`mnemo setup` writes client configs under $HOME, which the setup tests point at a temp
    directory, unless a client's own variable moves its config elsewhere. Drop those
    variables, so a developer who has one set never sees a test write their real config."""
    for var in ("KIMI_CODE_HOME",):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture(autouse=True)
def _isolate_global_logging_state():
    """configure_logging() mutates process-global state — the mnemo/llmkit loggers
    (handlers, level, propagate) and threading.excepthook. Snapshot and restore it around
    every test so a test that calls it can't leak (e.g. propagate=False, which breaks
    caplog, or a swapped excepthook) into the tests that run after it."""
    loggers = [logging.getLogger(name) for name in ("mnemo", "llmkit")]
    saved = [(logger, list(logger.handlers), logger.level, logger.propagate) for logger in loggers]
    saved_excepthook = threading.excepthook
    try:
        yield
    finally:
        for logger, handlers, level, propagate in saved:
            logger.handlers[:] = handlers
            logger.setLevel(level)
            logger.propagate = propagate
        threading.excepthook = saved_excepthook
