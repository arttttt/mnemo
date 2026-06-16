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

_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s"


def configure_logging() -> None:
    """Route the `mnemo` logger tree to stderr at MNEMO_LOG_LEVEL (default INFO).

    Idempotent: re-calling only refreshes the level (the handler is added once).
    `propagate` is disabled so each record is emitted once by our handler, not
    again by the root/uvicorn configuration.
    """
    logger = logging.getLogger("mnemo")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(os.environ.get("MNEMO_LOG_LEVEL", "INFO").upper())
    logger.propagate = False
