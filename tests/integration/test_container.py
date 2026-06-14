from mnemo.infrastructure.composition import build_container
from mnemo.infrastructure.config import Config


def test_container_wires_working_use_cases(tmp_path):
    config = Config(
        data_dir=str(tmp_path),
        embedder="hash",
        store="memory",
        store_path=str(tmp_path / "memory.json"),
    )
    container = build_container(config)

    stored = container.remember.execute(content="wired note", project="api")
    hits = container.search.execute(query="wired note", project="api")
    assert any(hit.id == stored.id for hit in hits)

    assert container.delete.purge().deleted == 1


def test_container_stamps_one_session_id_through_the_wiring(tmp_path):
    config = Config(
        data_dir=str(tmp_path),
        embedder="hash",
        store="memory",
        store_path=str(tmp_path / "memory.json"),
    )
    container = build_container(config)

    first = container.remember.execute(content="one", project="api")
    second = container.remember.execute(content="two", project="svc")  # different project, same run

    session_by_id = {memory.id: memory.session_id for memory in container.repository.list_all()}
    assert session_by_id[first.id] is not None
    assert session_by_id[first.id] == session_by_id[second.id]  # one run → one session
