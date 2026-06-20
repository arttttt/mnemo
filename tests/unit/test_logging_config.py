"""configure_logging() — the infrastructure logging setup."""
import logging
import threading

import pytest

from mnemo.infrastructure.logging_config import _log_thread_death, configure_logging


class _Capture(logging.Handler):
    """Collect records straight off a logger — independent of propagation, which
    configure_logging disables (so caplog via the root can't see mnemo records)."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


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


@pytest.fixture
def restore_threading_excepthook():
    # configure_logging installs a process-wide threading.excepthook (global state, like
    # sys.excepthook); snapshot and restore so it can't leak into other tests.
    saved = threading.excepthook
    try:
        yield
    finally:
        threading.excepthook = saved


def test_installs_a_thread_excepthook(restore_mnemo_logger, restore_threading_excepthook):
    configure_logging()

    assert threading.excepthook is _log_thread_death  # thread deaths route to the logger


def test_thread_excepthook_logs_critical_with_name_and_traceback(restore_threading_excepthook):
    capture = _Capture()
    thread_log = logging.getLogger("mnemo.thread")
    thread_log.addHandler(capture)
    try:
        threading.excepthook = _log_thread_death

        def boom():
            raise RuntimeError("boom")

        thread = threading.Thread(target=boom, name="mnemo-embed-test", daemon=True)
        thread.start()
        thread.join(2.0)
    finally:
        thread_log.removeHandler(capture)

    critical = [r for r in capture.records if r.levelno == logging.CRITICAL]
    assert critical, "a dying daemon thread must be logged at CRITICAL"
    assert "mnemo-embed-test" in critical[0].getMessage()
    assert critical[0].exc_info and critical[0].exc_info[0] is RuntimeError


def test_thread_excepthook_ignores_system_exit(restore_threading_excepthook):
    capture = _Capture()
    thread_log = logging.getLogger("mnemo.thread")
    thread_log.addHandler(capture)
    try:
        args = threading.ExceptHookArgs(
            (SystemExit, SystemExit(), None, threading.current_thread())
        )
        _log_thread_death(args)
    finally:
        thread_log.removeHandler(capture)

    assert capture.records == [], "SystemExit is a clean stop, not a crash"
