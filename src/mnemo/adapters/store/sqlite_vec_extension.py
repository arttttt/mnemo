"""Loading sqlite-vec, the SQLite extension the store keeps its vectors in.

Python's sqlite3 can load extensions only when Python was built with
--enable-loadable-sqlite-extensions; without it, a connection has no
enable_load_extension at all. Python from python.org, pyenv's default build and the
macOS system Python are built that way. On such a Python mnemo installs fine and then
fails as it opens the store, so the failure names the cause and the fix instead of
surfacing as an AttributeError.
"""
from __future__ import annotations

import sqlite3
import sys

import sqlite_vec


class SqliteExtensionsUnavailable(RuntimeError):
    pass


def load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec into `conn`, then turn extension loading off again."""
    try:
        enable_load_extension = conn.enable_load_extension
    except AttributeError:
        raise SqliteExtensionsUnavailable(
            f"the Python mnemo runs on ({sys.executable}) cannot load SQLite extensions, "
            "and mnemo keeps its vectors in one, sqlite-vec. Python from python.org, "
            "pyenv's default build and the macOS system Python lack this; the Python "
            "builds uv installs and Homebrew's have it. Install mnemo under such a Python, "
            "for example in a venv made with `uv venv --managed-python`; see the "
            "requirements in docs/13-usage.md."
        ) from None
    enable_load_extension(True)
    sqlite_vec.load(conn)
    enable_load_extension(False)
