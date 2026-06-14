"""The composition root wires a working use case from config, on each backend."""
import pytest

from mnemo.infrastructure.composition import build_container
from mnemo.infrastructure.config import Config


def test_from_env_defaults_to_sqlite(monkeypatch, tmp_path):
    for var in ("MNEMO_STORE", "MNEMO_SQLITE_PATH", "MNEMO_STORE_PATH", "MNEMO_EMBEDDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))

    config = Config.from_env()
    assert config.store == "sqlite"
    assert config.embedder == "fastembed"
    assert config.sqlite_path == str(tmp_path / "memory.db")
    assert config.store_path == str(tmp_path / "memory.json")


def test_build_container_wires_the_sqlite_backend(tmp_path):
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.sqlite_vec_repository import SqliteVecMemoryRepository

    config = Config(
        data_dir=str(tmp_path),
        embedder="hash",
        store="sqlite",
        store_path=str(tmp_path / "memory.json"),
        sqlite_path=str(tmp_path / "memory.db"),
    )
    container = build_container(config)
    assert isinstance(container.repository, SqliteVecMemoryRepository)

    stored = container.remember.execute(content="wired on sqlite", project="api")
    hits = container.search.execute(query="wired on sqlite", project="api")
    assert any(hit.id == stored.id for hit in hits)


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
