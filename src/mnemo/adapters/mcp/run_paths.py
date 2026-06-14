"""Filesystem layout for the on-demand service's runtime state (``~/.mnemo/run``).

A single place that derives the runtime directories from the config, so the
launcher (pidfile / spawn lock), the service (idle sweep) and the connector
(presence marker) all agree on where things live.
"""
from __future__ import annotations

from pathlib import Path

from mnemo.infrastructure.config import Config


def run_dir(config: Config) -> Path:
    """The runtime directory that sits next to the data dir (e.g. ``~/.mnemo/run``)."""
    return Path(config.data_dir).expanduser().parent / "run"


def connectors_dir(config: Config) -> Path:
    """Where each live connector keeps its presence marker (one flock per run)."""
    return run_dir(config) / "connectors"
