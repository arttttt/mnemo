"""The composition root wires a working use case from config, on each backend."""
import pytest

from mnemo.infrastructure.composition import build_container
from mnemo.infrastructure.config import Config


def test_from_env_defaults_to_lancedb(monkeypatch, tmp_path):
    for var in ("MNEMO_STORE", "MNEMO_LANCEDB_URI", "MNEMO_STORE_PATH", "MNEMO_EMBEDDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))

    config = Config.from_env()
    assert config.store == "lancedb"
    assert config.embedder == "fastembed"
    assert config.lancedb_uri == str(tmp_path / "memory")
    assert config.store_path == str(tmp_path / "memory.json")


@pytest.mark.heavy
def test_build_container_wires_the_lancedb_backend(tmp_path):
    pytest.importorskip("lancedb")
    from mnemo.adapters.store.lancedb_repository import LanceDbMemoryRepository

    config = Config(
        data_dir=str(tmp_path),
        embedder="hash",
        store="lancedb",
        store_path=str(tmp_path / "memory.json"),
        lancedb_uri=str(tmp_path / "memory"),
    )
    container = build_container(config)
    assert isinstance(container.repository, LanceDbMemoryRepository)

    stored = container.remember.execute(content="wired on lancedb", project="api")
    hits = container.search.execute(query="wired on lancedb", project="api")
    assert any(hit.id == stored.id for hit in hits)
