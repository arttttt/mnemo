"""The composition root wires a working use case from config, on each backend."""
import pytest

from mnemo.infrastructure.composition import build_container
from mnemo.infrastructure.config import Config


def test_from_env_defaults(monkeypatch, tmp_path):
    for var in ("MNEMO_SQLITE_PATH", "MNEMO_EMBEDDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))

    config = Config.from_env()
    assert config.embedder == "pplx"  # the default embedder
    assert config.sqlite_path == str(tmp_path / "memory.db")


def test_build_container_wires_the_sqlite_backend(tmp_path):
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.sqlite_vec_repository import SqliteRepositoryImpl

    config = Config(
        data_dir=str(tmp_path),
        embedder="hash",
        sqlite_path=str(tmp_path / "memory.db"),
    )
    container = build_container(config)
    assert isinstance(container.repository, SqliteRepositoryImpl)

    container.create_project.execute("api")
    stored = container.remember.execute(content="wired on sqlite", project="api")
    hits = container.search.execute(query="wired on sqlite", project="api")
    assert any(hit.id == stored.id for hit in hits)
