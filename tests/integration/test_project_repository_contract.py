"""Project registry contract, exercised against the SQLite backend (the sole store)."""
import pytest

from mnemo.domain.constants import GLOBAL_PROJECT
from mnemo.domain.project import Project


def _sqlite(tmp_path):
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.sqlite_connections import SqliteConnections
    from mnemo.adapters.store.sqlite_project_repository import (
        SqliteProjectRepositoryImpl,
    )

    return SqliteProjectRepositoryImpl(SqliteConnections(str(tmp_path / "memory.db")))


@pytest.fixture
def projects(tmp_path):
    return _sqlite(tmp_path)


def test_create_then_exists_and_get(projects):
    assert projects.exists("pa-kmp") is False
    projects.create(Project.create("pa-kmp", "Personal assistant, KMP"))

    assert projects.exists("pa-kmp") is True
    got = projects.get("pa-kmp")
    assert got is not None
    assert (got.slug, got.description) == ("pa-kmp", "Personal assistant, KMP")


def test_get_unknown_returns_none(projects):
    assert projects.get("nope") is None


def test_description_is_optional(projects):
    projects.create(Project.create("bare"))
    assert projects.get("bare").description is None


def test_update_description(projects):
    projects.create(Project.create("p"))
    projects.update_description("p", "now described")
    assert projects.get("p").description == "now described"


def test_delete_removes_the_row(projects):
    projects.create(Project.create("temp"))
    projects.delete("temp")
    assert projects.exists("temp") is False


def test_list_all_lists_registered_and_excludes_global(projects):
    projects.create(Project.create("first"))
    projects.create(Project.create("second"))
    slugs = {p.slug for p in projects.list_all()}
    assert slugs == {"first", "second"}  # order is recency, not asserted (timestamp ties)


def test_global_sentinel_is_seeded_but_hidden(projects):
    # The reserved row exists (FK integrity for global memories) but is not a project.
    assert projects.exists(GLOBAL_PROJECT) is True
    assert all(p.slug != GLOBAL_PROJECT for p in projects.list_all())


def test_delete_all_wipes_and_reseeds_the_global_sentinel(projects):
    projects.create(Project.create("first"))
    projects.create(Project.create("second"))

    projects.delete_all()

    assert projects.list_all() == []                # every real project gone
    assert projects.exists(GLOBAL_PROJECT) is True  # the sentinel is re-seeded, registry usable
