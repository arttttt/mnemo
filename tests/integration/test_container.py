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
