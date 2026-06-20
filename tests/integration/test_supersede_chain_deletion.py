"""Deletion vs the supersede / topic_key chain — characterizes the CURRENT bugs.

A topic_key chain is `v1 <- v2 <- v3` (v3 active, v1/v2 superseded): each successor
carries `supersedes = prior.id` and a typed `Link.supersedes(successor -> prior)`.

These tests assert the DESIRED behavior and are marked xfail(strict) — they fail on
today's code (so they "catch" the bugs) while keeping the suite green. When the fix
lands, drop the xfail markers and they become guards.
"""
import pytest

pytest.importorskip("sqlite_vec")

from mnemo.adapters.embedding.hash_embedder import HashEmbedder
from mnemo.domain.link import Link
from mnemo.domain.memory import Memory
from tests.support.sqlite_store import open_store

_TOPIC = "auth/model"
_PROJECT = "api"


def _supersede(repo, embedder, prior, content):
    """Evolve `prior` via the production supersede path (prior -> superseded, new active)."""
    successor = Memory.create(content, project=_PROJECT, topic_key=_TOPIC)
    successor.supersedes = prior.id
    link = Link.supersedes(source_id=successor.id, target_id=prior.id, provenance=_TOPIC)
    repo.supersede(successor, link, embedder.encode(content))
    return successor


def _chain(repo, embedder):
    """Build v1 <- v2 <- v3 (v3 active). Returns (v1, v2, v3)."""
    v1 = Memory.create("auth model v1", project=_PROJECT, topic_key=_TOPIC)
    repo.add(v1, embedder.encode(v1.content))
    v2 = _supersede(repo, embedder, v1, "auth model v2")
    v3 = _supersede(repo, embedder, v2, "auth model v3")
    return v1, v2, v3


@pytest.mark.xfail(
    strict=True,
    reason="BUG: deleting the active head strands the topic_key — the prior is NOT "
    "promoted to active, so find_active_by_topic_key returns None and the history "
    "becomes unreachable / the next upsert silently forks.",
)
def test_deleting_active_head_promotes_the_prior(tmp_path):
    embedder = HashEmbedder()
    repo, _ = open_store(tmp_path, embedder.dim, projects=(_PROJECT,))
    v1, v2, v3 = _chain(repo, embedder)

    repo.delete([v3.id])  # remove the active head

    active = repo.find_active_by_topic_key(_TOPIC, _PROJECT)
    assert active is not None, "topic_key has no active record after deleting its head"
    assert active.id == v2.id, "the immediate prior should be promoted to active"


@pytest.mark.xfail(
    strict=True,
    reason="BUG: the `supersedes` column has no FK, so deleting a referenced memory "
    "leaves a dangling pointer on the survivor (only the links edge cascades).",
)
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
