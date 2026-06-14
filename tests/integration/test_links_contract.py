"""Typed-links contract, run against every links-capable backend.

In-memory and SQLite both implement the `links` table. The legacy LanceDB
backend deliberately does not (it is migration-source only), so it is excluded
here.
"""
import pytest

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.domain.link import Link
from mnemo.domain.link_type import LinkType
from mnemo.domain.memory import Memory


def _in_memory(tmp_path):
    from mnemo.adapters.store.in_memory_repository import InMemoryMemoryRepository

    return InMemoryMemoryRepository(path=str(tmp_path / "memory.json"))


def _sqlite(tmp_path):
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.sqlite_vec_repository import SqliteVecMemoryRepository

    return SqliteVecMemoryRepository(path=str(tmp_path / "memory.db"))


@pytest.fixture(
    params=[
        pytest.param(_in_memory, id="in_memory"),
        pytest.param(_sqlite, id="sqlite"),
    ]
)
def open_repo(request, tmp_path):
    """Return a zero-arg factory that (re)opens a repo at one fixed location."""
    return lambda: request.param(tmp_path)


def test_link_is_retrievable_from_both_endpoints(open_repo):
    repo = open_repo()
    repo.add_link(
        Link.supersedes(source_id="new", target_id="old", provenance="auth/model")
    )

    forward = repo.links_for("new")
    assert len(forward) == 1
    link = forward[0]
    assert (link.source_id, link.target_id, link.type) == ("new", "old", LinkType.SUPERSEDES)
    assert link.provenance == "auth/model"
    assert repo.links_for("old") == forward  # reachable from the target too
    assert repo.links_for("unrelated") == []


def test_links_persist_across_reopen(open_repo):
    open_repo().add_link(
        Link.supersedes(source_id="a", target_id="b", provenance="topic")
    )

    reopened = open_repo()
    persisted = reopened.links_for("a")
    assert [(link.target_id, link.provenance) for link in persisted] == [("b", "topic")]


def test_deleting_an_endpoint_removes_its_link(open_repo):
    repo = open_repo()
    embedder = HashEmbedder()
    old = Memory.create("auth model v1", project="api")
    new = Memory.create("auth model v2", project="api")
    repo.add(old, embedder.encode(old.content))
    repo.add(new, embedder.encode(new.content))
    repo.add_link(
        Link.supersedes(source_id=new.id, target_id=old.id, provenance="auth/model")
    )

    assert repo.delete([old.id]) == 1
    assert repo.links_for(new.id) == []  # the edge no longer dangles
