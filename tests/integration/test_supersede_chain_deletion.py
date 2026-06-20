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


def _chain5(repo, embedder):
    """Build v1 <- v2 <- v3 <- v4 <- v5 (v5 active). Returns (v1, v2, v3, v4, v5)."""
    v1 = Memory.create("auth model v1", project=_PROJECT, topic_key=_TOPIC)
    repo.add(v1, embedder.encode(v1.content))
    v2 = _supersede(repo, embedder, v1, "auth model v2")
    v3 = _supersede(repo, embedder, v2, "auth model v3")
    v4 = _supersede(repo, embedder, v3, "auth model v4")
    v5 = _supersede(repo, embedder, v4, "auth model v5")
    return v1, v2, v3, v4, v5


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


def test_deleting_several_members_at_once_promotes_and_splices(tmp_path):
    # One delete() may remove several members of the same chain. Deleting the head AND an
    # interior node together must still promote the oldest survivor and leave no dangling
    # pointer — the splice walks past every deleted ancestor in one pass, not just one hop.
    embedder = HashEmbedder()
    repo, _ = open_store(tmp_path, embedder.dim, projects=(_PROJECT,))
    v1, v2, v3 = _chain(repo, embedder)

    repo.delete([v2.id, v3.id])  # head + interior in a single call

    active = repo.find_active_by_topic_key(_TOPIC, _PROJECT)
    assert active is not None, "topic_key has no active record after a batch delete"
    assert active.id == v1.id, "the sole survivor should be promoted to active"
    survivors = {m.id for m in repo.list_all()}
    dangling = [
        m.id for m in repo.list_all()
        if m.supersedes is not None and m.supersedes not in survivors
    ]
    assert dangling == [], f"memories with a dangling supersedes pointer: {dangling}"


def test_deleting_the_root_repoints_its_successor_to_nothing(tmp_path):
    # Deleting the oldest member (whose supersedes is None) must null the next member's
    # pointer rather than strand it, and the active head is untouched.
    embedder = HashEmbedder()
    repo, _ = open_store(tmp_path, embedder.dim, projects=(_PROJECT,))
    v1, v2, v3 = _chain(repo, embedder)

    repo.delete([v1.id])  # the root of the chain

    by_id = {m.id: m for m in repo.list_all()}
    assert by_id[v2.id].supersedes is None, "the successor of the root should point at nothing"
    assert by_id[v3.id].supersedes == v2.id, "the rest of the lineage is preserved"
    active = repo.find_active_by_topic_key(_TOPIC, _PROJECT)
    assert active.id == v3.id, "deleting a superseded root must not move the active head"


def test_deleting_the_whole_chain_retires_the_topic_key(tmp_path):
    # Removing every member leaves nothing to promote — the topic_key is simply gone.
    embedder = HashEmbedder()
    repo, _ = open_store(tmp_path, embedder.dim, projects=(_PROJECT,))
    v1, v2, v3 = _chain(repo, embedder)

    assert repo.delete([v1.id, v2.id, v3.id]) == 3

    assert repo.find_active_by_topic_key(_TOPIC, _PROJECT) is None
    assert list(repo.list_all()) == []


def test_deleting_consecutive_interior_members_splices_across_the_whole_gap(tmp_path):
    # Several ADJACENT interior nodes deleted at once: the surviving head must repoint past
    # the entire run in one pass (the splice walk hops v4 -> v3 -> v2 -> v1), not one node.
    embedder = HashEmbedder()
    repo, _ = open_store(tmp_path, embedder.dim, projects=(_PROJECT,))
    v1, v2, v3, v4, v5 = _chain5(repo, embedder)

    repo.delete([v2.id, v3.id, v4.id])  # three consecutive interior nodes

    by_id = {m.id: m for m in repo.list_all()}
    assert set(by_id) == {v1.id, v5.id}, "only the root and the head should survive"
    assert by_id[v5.id].supersedes == v1.id, "the head should splice past the whole deleted run"
    active = repo.find_active_by_topic_key(_TOPIC, _PROJECT)
    assert active.id == v5.id, "deleting interior nodes must not move the active head"


def test_deleting_alternating_members_splices_each_gap_independently(tmp_path):
    # Non-adjacent interior deletes ("every other"): each survivor repoints past its own
    # deleted predecessor, so one delete() performs two independent splices.
    embedder = HashEmbedder()
    repo, _ = open_store(tmp_path, embedder.dim, projects=(_PROJECT,))
    v1, v2, v3, v4, v5 = _chain5(repo, embedder)

    repo.delete([v2.id, v4.id])  # delete every other interior node

    by_id = {m.id: m for m in repo.list_all()}
    assert set(by_id) == {v1.id, v3.id, v5.id}
    assert by_id[v3.id].supersedes == v1.id, "v3 should splice past the deleted v2 to v1"
    assert by_id[v5.id].supersedes == v3.id, "v5 should splice past the deleted v4 to v3"
    active = repo.find_active_by_topic_key(_TOPIC, _PROJECT)
    assert active.id == v5.id
