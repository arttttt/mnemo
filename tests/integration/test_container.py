from mnemo.infrastructure.config import Config
from mnemo.infrastructure.container import build_container


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
