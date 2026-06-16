"""configure_logging() — the infrastructure logging setup."""
import logging

import pytest

from mnemo.infrastructure.logging_config import configure_logging


@pytest.fixture
def restore_mnemo_logger():
    # configure_logging mutates the global "mnemo" logger (handlers, level, propagate);
    # snapshot and restore so it cannot leak into other tests — notably caplog, which
    # relies on propagation staying enabled.
    logger = logging.getLogger("mnemo")
    saved = (list(logger.handlers), logger.level, logger.propagate)
    logger.handlers[:] = []  # clean slate for deterministic assertions
    try:
        yield logger
    finally:
        logger.handlers[:] = saved[0]
        logger.setLevel(saved[1])
        logger.propagate = saved[2]


def test_attaches_handler_defaults_to_info_and_stops_propagation(
    restore_mnemo_logger, monkeypatch
):
    monkeypatch.delenv("MNEMO_LOG_LEVEL", raising=False)
    logger = restore_mnemo_logger

    configure_logging()

    assert logger.handlers, "a handler must be attached so mnemo logs are emitted"
    assert logger.level == logging.INFO  # the default level
    assert logger.propagate is False     # emitted once by our handler, not again via root


def test_is_idempotent_and_honours_env_level(restore_mnemo_logger, monkeypatch):
    monkeypatch.setenv("MNEMO_LOG_LEVEL", "DEBUG")
    logger = restore_mnemo_logger

    configure_logging()
    configure_logging()  # a second call must not stack a second handler

    assert len(logger.handlers) == 1
    assert logger.level == logging.DEBUG
