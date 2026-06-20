import pytest

from mnemo.infrastructure.composition import build_container
from mnemo.infrastructure.config import Config

pytest.importorskip("sqlite_vec")


def test_container_wires_working_use_cases(tmp_path):
    config = Config(
        data_dir=str(tmp_path),
        embedder="hash",
        sqlite_path=str(tmp_path / "memory.db"),
    )
    container = build_container(config)

    container.create_project.execute("api")
    stored = container.remember.execute(content="wired note", project="api")
    hits = container.search.execute(query="wired note", project="api")
    assert any(hit.id == stored.id for hit in hits)

    assert container.delete.purge().deleted == 1


def test_container_stamps_one_session_id_through_the_wiring(tmp_path):
    config = Config(
        data_dir=str(tmp_path),
        embedder="hash",
        sqlite_path=str(tmp_path / "memory.db"),
    )
    container = build_container(config)

    container.create_project.execute("api")
    container.create_project.execute("svc")
    first = container.remember.execute(content="one", project="api")
    second = container.remember.execute(content="two", project="svc")  # different project, same run

    session_by_id = {memory.id: memory.session_id for memory in container.repository.list_all()}
    assert session_by_id[first.id] is not None
    assert session_by_id[first.id] == session_by_id[second.id]  # one run → one session


def test_delete_project_cascades_through_the_wiring(tmp_path):
    config = Config(
        data_dir=str(tmp_path),
        embedder="hash",
        sqlite_path=str(tmp_path / "memory.db"),
    )
    container = build_container(config)
    container.create_project.execute("api")
    container.create_project.execute("other")
    container.remember.execute(content="doomed note", project="api")
    kept = container.remember.execute(content="survivor note", project="other")

    deleted = container.delete_project.execute("api")

    assert deleted.slug == "api"
    assert container.projects.exists("api") is False
    # the api memory cascaded away with the project; the other project is untouched
    assert {m.id for m in container.repository.list_all()} == {kept.id}
