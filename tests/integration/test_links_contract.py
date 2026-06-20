"""Typed-links contract, exercised against the SQLite backend (the sole store)."""
import pytest

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.domain.link import Link
from mnemo.domain.link_type import LinkType
from mnemo.domain.memory import Memory
from tests.support.sqlite_store import open_store


def _sqlite(tmp_path):
    # Links foreign-key to memories(id), so the endpoints must be real rows;
    # register the project so those memory inserts satisfy their own FK.
    repo, _ = open_store(tmp_path, HashEmbedder().dim, projects=("api",))
    return repo


@pytest.fixture
def open_repo(tmp_path):
    """Return a zero-arg factory that (re)opens a repo at one fixed location."""
    return lambda: _sqlite(tmp_path)


def _add(repo, content):
    memory = Memory.create(content, project="api")
    repo.add(memory, HashEmbedder().encode(content))
    return memory


def test_link_is_retrievable_from_both_endpoints(open_repo):
    repo = open_repo()
    new = _add(repo, "auth model v2")
    old = _add(repo, "auth model v1")
    repo.add_link(
        Link.supersedes(source_id=new.id, target_id=old.id, provenance="auth/model")
    )

    forward = repo.links_for(new.id)
    assert len(forward) == 1
    link = forward[0]
    assert (link.source_id, link.target_id, link.type) == (new.id, old.id, LinkType.SUPERSEDES)
    assert link.provenance == "auth/model"
    assert repo.links_for(old.id) == forward  # reachable from the target too
    assert repo.links_for("unrelated") == []


def test_links_persist_across_reopen(open_repo):
    first = open_repo()
    a = _add(first, "auth model v2")
    b = _add(first, "auth model v1")
    first.add_link(Link.supersedes(source_id=a.id, target_id=b.id, provenance="topic"))

    reopened = open_repo()
    persisted = reopened.links_for(a.id)
    assert [(link.target_id, link.provenance) for link in persisted] == [(b.id, "topic")]


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
