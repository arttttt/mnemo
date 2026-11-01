"""The installed mnemo distribution's version, read from its package metadata.

One lookup shared by every surface that reports the version — the CLI's `version`
command and the MCP server's handshake — so they can never disagree.
"""
from __future__ import annotations

from importlib.metadata import version

_DISTRIBUTION = "mnemo"


def installed_version() -> str:
    return version(_DISTRIBUTION)
