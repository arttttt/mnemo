"""Process-wide setup for mnemo's own loggers — a cross-cutting infrastructure
concern, kept out of the use cases and adapters.

An entry point (the shared service) calls configure_logging() once at start-up.
The format carries the thread name so the background embed worker (`mnemo-embed-N`)
is visibly distinct from the request thread — that is how one tells the deferred
encode from any inline (backpressure) work in the logs.
"""
from __future__ import annotations

import logging
import os
import threading

_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s"

# A dedicated child of "mnemo" so thread-death records are filterable; it inherits the
# handler/level/propagate set on the parent in configure_logging().
_thread_log = logging.getLogger("mnemo.thread")


def _log_thread_death(args: threading.ExceptHookArgs) -> None:
    """Route an uncaught exception that killed a thread into the mnemo logger.

    The stdlib default threading.excepthook only prints a raw traceback to stderr, so a
    dying daemon (the embed worker, the idle monitor) vanishes silently while a core
    function stays disabled for the whole process — no app-level signal, no recovery.
    Log it once at CRITICAL with the thread name and traceback so the death is loud.
    A SystemExit is a clean stop, not a crash (as in the stdlib default) — ignore it.
    """
    if args.exc_type is SystemExit:
        return
    thread = args.thread
    _thread_log.critical(
        "daemon thread %r died on an unhandled exception — a background function is now "
        "disabled for the process lifetime",
        thread.name if thread is not None else "<unknown>",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def configure_logging() -> None:
    """Route the `mnemo` and `llmkit` logger trees to stderr at MNEMO_LOG_LEVEL (default INFO).

    Idempotent: re-calling only refreshes the level (the handler is added once).
    `propagate` is disabled so each record is emitted once by our handler, not again by
    the root/uvicorn configuration. `llmkit` is mnemo's inference package, so its load/run
    logs (model timing + RSS) surface through here too.
    """
    level = os.environ.get("MNEMO_LOG_LEVEL", "INFO").upper()
    for name in ("mnemo", "llmkit"):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(_FORMAT))
            logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    # Also route any exception that kills a (daemon) thread into the mnemo logger instead
    # of the stdlib default's raw-stderr traceback. Idempotent: the same callable each call.
    threading.excepthook = _log_thread_death
