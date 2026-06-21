"""Shared test setup."""
import logging
import threading

import pytest


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
