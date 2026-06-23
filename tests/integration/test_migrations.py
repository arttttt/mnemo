"""The disposable project-FK migration upgrades a pre-FK store in place.

Builds a realistic pre-FK store (the current schema minus the foreign keys), runs the
migration, and asserts the FKs were added, data preserved, the registry seeded, and
that it is idempotent.
"""
import sqlite3

import pytest

pytest.importorskip("sqlite_vec")

import sqlite_vec
from sqlite_vec import serialize_float32

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.adapters.store.sqlite_connections import SqliteConnections
from mnemo.adapters.store.sqlite_project_repository import SqliteProjectRepositoryImpl
from mnemo.adapters.store.sqlite_vec_repository import (
    SqliteRepositoryImpl,
    _FTS_STATEMENT,
    _INDEX_STATEMENTS,
    _TRIGGER_STATEMENTS,
    _create_table_sql,
)
from mnemo.domain.memory import Memory
from mnemo.infrastructure.migrations import (
    add_project_foreign_keys,
    drop_links_table,
    remap_retired_memory_types,
)

_DIM = 256
# The pre-FK memories DDL = today's schema with the project foreign key stripped.
_OLD_MEMORIES = _create_table_sql("memories", _DIM).replace(
    " REFERENCES projects(slug) ON DELETE CASCADE", ""
)
_OLD_LINKS = (
    "CREATE TABLE links (source_id TEXT NOT NULL, target_id TEXT NOT NULL,"
    " type TEXT NOT NULL, provenance TEXT NOT NULL, created_at TEXT NOT NULL,"
    " PRIMARY KEY (source_id, target_id, type))"
)
_TS = "2026-01-01T00:00:00+00:00"


def _insert_memory(conn, embedder, mid, content, project, scope="project", mtype="decision"):
    conn.execute(
        "INSERT INTO memories (id, content, embedding, type, scope, project, tags,"
        " related_files, topic_key, session_id, status, supersedes, hash, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (mid, content, serialize_float32(list(embedder.encode(content))), mtype,
         scope, project, "[]", "[]", None, None, "active", None, mid + "-hash", _TS, _TS),
    )


def _build_pre_fk_store(path, embedder):
    """A pre-FK store: no `projects` table, memories/links without foreign keys."""
    conn = sqlite3.connect(path)
    conn.isolation_level = None
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(_OLD_MEMORIES)
    conn.execute(_OLD_LINKS)
    for statement in _INDEX_STATEMENTS:
        conn.execute(statement)
    conn.execute(_FTS_STATEMENT)
    for statement in _TRIGGER_STATEMENTS:
        conn.execute(statement)
    _insert_memory(conn, embedder, "m-api", "auth model with zzqqx token", "api")
    _insert_memory(conn, embedder, "m-other", "redis cache eviction", "other")
    _insert_memory(conn, embedder, "m-global", "always confirm destructive ops", "__global__", scope="global")
    conn.execute(
        "INSERT INTO links (source_id, target_id, type, provenance, created_at)"
        " VALUES ('m-api', 'm-other', 'supersedes', 'topic', ?)",
        (_TS,),
    )
    conn.close()


def test_migration_adds_fks_seeds_registry_and_preserves_data(tmp_path):
    path = str(tmp_path / "memory.db")
    embedder = HashEmbedder(dim=_DIM)
    _build_pre_fk_store(path, embedder)

    assert add_project_foreign_keys(path) is True

    conns = SqliteConnections(path)
    projects = SqliteProjectRepositoryImpl(conns)
    repo = SqliteRepositoryImpl(conns, _DIM)

    # registry seeded from the distinct project values (global stays hidden)
    assert {p.slug for p in projects.list_all()} == {"api", "other"}
    assert projects.exists("__global__") is True

    # memories preserved, embeddings kept (not reset to pending)
    assert {m.id for m in repo.list_all()} == {"m-api", "m-other", "m-global"}
    assert repo.pending_count() == 0
    assert repo.has_vector("m-api") is True
    # FTS index was rebuilt — a still-stored token is lexically findable
    from mnemo.application.retrieval import Retrieval
    from mnemo.application.search_criteria import SearchCriteria
    hits = repo.retrieve(Retrieval(criteria=SearchCriteria(scope="all"), limit=5, text="zzqqx",
                                   vector=embedder.encode("zzqqx")))
    assert any("zzqqx" in h.memory.content for h in hits)


def test_migration_makes_the_fk_enforced(tmp_path):
    path = str(tmp_path / "memory.db")
    embedder = HashEmbedder(dim=_DIM)
    _build_pre_fk_store(path, embedder)
    add_project_foreign_keys(path)

    repo = SqliteRepositoryImpl(SqliteConnections(path), _DIM)
    with pytest.raises(sqlite3.IntegrityError):
        repo.add(Memory.create("ghost", project="ghost"), embedder.encode("ghost"))


def test_migration_cascades_after_upgrade(tmp_path):
    path = str(tmp_path / "memory.db")
    embedder = HashEmbedder(dim=_DIM)
    _build_pre_fk_store(path, embedder)
    add_project_foreign_keys(path)

    conns = SqliteConnections(path)
    projects = SqliteProjectRepositoryImpl(conns)
    repo = SqliteRepositoryImpl(conns, _DIM)

    projects.delete("api")  # the new FK cascades: api's memories go away
    assert {m.id for m in repo.list_all()} == {"m-other", "m-global"}


def test_migration_is_idempotent(tmp_path):
    path = str(tmp_path / "memory.db")
    _build_pre_fk_store(path, HashEmbedder(dim=_DIM))

    assert add_project_foreign_keys(path) is True   # first run migrates
    assert add_project_foreign_keys(path) is False  # second run is a no-op


def test_migration_noop_when_store_absent(tmp_path):
    assert add_project_foreign_keys(str(tmp_path / "does-not-exist.db")) is False


def test_drop_links_table_removes_it_and_is_idempotent(tmp_path):
    path = str(tmp_path / "memory.db")
    _build_pre_fk_store(path, HashEmbedder(dim=_DIM))  # has a links table

    assert drop_links_table(path) is True  # dropped

    conn = sqlite3.connect(path)
    present = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='links'"
    ).fetchone()
    conn.close()
    assert present is None

    assert drop_links_table(path) is False  # idempotent no-op
    assert drop_links_table(str(tmp_path / "absent.db")) is False  # no store → no-op


def _store_with_memory_schema(path):
    """A store with just the memories schema + derived state (FTS/triggers/indexes), no FK —
    enough to insert raw rows of any type and run a column-value migration over them."""
    conn = sqlite3.connect(path)
    conn.isolation_level = None
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(_OLD_MEMORIES)
    for statement in _INDEX_STATEMENTS:
        conn.execute(statement)
    conn.execute(_FTS_STATEMENT)
    for statement in _TRIGGER_STATEMENTS:
        conn.execute(statement)
    return conn


def test_remap_retired_types_rehomes_and_is_idempotent(tmp_path):
    path = str(tmp_path / "memory.db")
    embedder = HashEmbedder(dim=_DIM)
    conn = _store_with_memory_schema(path)
    for mid, mtype in [
        ("m-debug", "debug"), ("m-design", "design"), ("m-feature", "feature"),
        ("m-snip", "code-snippet"), ("m-disc", "discussion"), ("m-keep", "research"),
    ]:
        _insert_memory(conn, embedder, mid, "note " + mid, "api", mtype=mtype)
    conn.close()

    assert remap_retired_memory_types(path) is True

    conn = sqlite3.connect(path)
    kinds = dict(conn.execute("SELECT id, type FROM memories"))
    conn.close()
    assert kinds == {
        "m-debug": "learning", "m-design": "decision", "m-feature": "decision",
        "m-snip": "working-notes", "m-disc": "working-notes", "m-keep": "research",
    }

    assert remap_retired_memory_types(path) is False  # idempotent — nothing left to remap
    assert remap_retired_memory_types(str(tmp_path / "absent.db")) is False  # no store → no-op
