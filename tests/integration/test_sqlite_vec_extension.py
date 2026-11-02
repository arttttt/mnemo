"""Loading sqlite-vec into the store's connections, and failing clearly on a Python that
cannot load SQLite extensions (built without --enable-loadable-sqlite-extensions)."""
import sqlite3
import sys

import pytest

pytest.importorskip("sqlite_vec")

from mnemo.adapters.store.sqlite_connections import SqliteConnections
from mnemo.adapters.store.sqlite_vec_extension import (
    SqliteExtensionsUnavailable,
    load_sqlite_vec,
)
from mnemo.infrastructure.migrations import (
    add_project_foreign_keys,
    drop_links_table,
    remap_retired_memory_types,
)


class _WithoutExtensionLoading:
    """A connection as such a Python makes it: no enable_load_extension at all."""

    isolation_level = None
    row_factory = None

    def execute(self, *args):
        raise AssertionError("the store went on without its extension")

    def close(self):
        pass


def test_sqlite_vec_is_loaded_and_extension_loading_turned_off_again():
    conn = sqlite3.connect(":memory:")

    load_sqlite_vec(conn)

    assert conn.execute("SELECT vec_version()").fetchone()[0].startswith("v")
    with pytest.raises(sqlite3.OperationalError, match="not authorized"):
        conn.load_extension("anything")  # with loading on, it would fail to find the file


def test_a_python_without_extension_loading_is_named_with_the_fix():
    with pytest.raises(SqliteExtensionsUnavailable) as failure:
        load_sqlite_vec(_WithoutExtensionLoading())

    message = str(failure.value)
    assert sys.executable in message
    assert "cannot load SQLite extensions" in message
    assert "uv venv --managed-python" in message


@pytest.mark.parametrize("open_store", [
    lambda path: SqliteConnections(str(path)),  # every run
    # The migrations, which open a store that already exists before anything else does.
    add_project_foreign_keys,
    drop_links_table,
    remap_retired_memory_types,
])
def test_the_store_fails_clearly_without_extension_loading(tmp_path, monkeypatch, open_store):
    path = tmp_path / "memory.db"
    path.touch()
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: _WithoutExtensionLoading())

    with pytest.raises(SqliteExtensionsUnavailable):
        open_store(path)
