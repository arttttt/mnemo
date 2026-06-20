"""Deletion keeps the supersede / topic_key chain consistent.

A topic_key chain is `v1 <- v2 <- v3` (v3 active, v1/v2 superseded): each successor
carries `supersedes = prior.id` (the chain's single encoding).

Deleting a member must not strand the chain: deleting the active head promotes the
prior to active, and deleting an interior node splices the pointer so nothing dangles.
"""
import pytest

pytest.importorskip("sqlite_vec")

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.domain.memory import Memory
from tests.support.sqlite_store import open_store

_TOPIC = "auth/model"
_PROJECT = "api"


def _supersede(repo, embedder, prior, content):
    """Evolve `prior` via the production supersede path (prior -> superseded, new active)."""
    successor = Memory.create(content, project=_PROJECT, topic_key=_TOPIC)
    successor.supersedes = prior.id
    repo.supersede(successor, embedder.encode(content))
    return successor


def _chain(repo, embedder):
    """Build v1 <- v2 <- v3 (v3 active). Returns (v1, v2, v3)."""
    v1 = Memory.create("auth model v1", project=_PROJECT, topic_key=_TOPIC)
    repo.add(v1, embedder.encode(v1.content))
    v2 = _supersede(repo, embedder, v1, "auth model v2")
    v3 = _supersede(repo, embedder, v2, "auth model v3")
    return v1, v2, v3


def test_deleting_active_head_promotes_the_prior(tmp_path):
    embedder = HashEmbedder()
    repo, _ = open_store(tmp_path, embedder.dim, projects=(_PROJECT,))
    v1, v2, v3 = _chain(repo, embedder)

    repo.delete([v3.id])  # remove the active head

    active = repo.find_active_by_topic_key(_TOPIC, _PROJECT)
    assert active is not None, "topic_key has no active record after deleting its head"
    assert active.id == v2.id, "the immediate prior should be promoted to active"


def test_deleting_interior_member_leaves_no_dangling_supersedes(tmp_path):
    embedder = HashEmbedder()
    repo, _ = open_store(tmp_path, embedder.dim, projects=(_PROJECT,))
    v1, v2, v3 = _chain(repo, embedder)

    repo.delete([v2.id])  # remove the middle (superseded) node

    survivors = {m.id for m in repo.list_all()}
    dangling = [
        m.id for m in repo.list_all()
        if m.supersedes is not None and m.supersedes not in survivors
    ]
    assert dangling == [], f"memories with a dangling supersedes pointer: {dangling}"
