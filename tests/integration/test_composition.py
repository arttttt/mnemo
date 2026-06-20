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
    assert config.embedder == "pplx"  # the default embedder
    assert config.sqlite_path == str(tmp_path / "memory.db")
    assert config.store_path == str(tmp_path / "memory.json")


def test_from_env_reads_embed_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MNEMO_EMBED_MODEL", "BAAI/bge-m3")
    assert Config.from_env().embed_model == "BAAI/bge-m3"
    monkeypatch.delenv("MNEMO_EMBED_MODEL")
    assert Config.from_env().embed_model is None


def test_build_embedder_forwards_configured_model(monkeypatch):
    """MNEMO_EMBED_MODEL must reach FastEmbedEmbedder(model_name=...), without loading it."""
    import mnemo.adapters.embedding.fastembed_embedder as fe
    from mnemo.infrastructure.composition import _build_embedder

    captured = {}

    class StubFastEmbed:
        def __init__(self, model_name=fe.DEFAULT_MODEL):
            captured["model_name"] = model_name

    monkeypatch.setattr(fe, "FastEmbedEmbedder", StubFastEmbed)

    def _config(embed_model):
        return Config(
            data_dir="/tmp", embedder="fastembed", store="memory",
            store_path="/tmp/m.json", embed_model=embed_model,
        )

    _build_embedder(_config("BAAI/bge-m3"))
    assert captured["model_name"] == "BAAI/bge-m3"

    _build_embedder(_config(None))  # omitted -> adapter default
    assert captured["model_name"] == fe.DEFAULT_MODEL


def test_build_container_wires_the_sqlite_backend(tmp_path):
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.sqlite_vec_repository import SqliteRepositoryImpl

    config = Config(
        data_dir=str(tmp_path),
        embedder="hash",
        store="sqlite",
        store_path=str(tmp_path / "memory.json"),
        sqlite_path=str(tmp_path / "memory.db"),
    )
    container = build_container(config)
    assert isinstance(container.repository, SqliteRepositoryImpl)

    stored = container.remember.execute(content="wired on sqlite", project="api")
    hits = container.search.execute(query="wired on sqlite", project="api")
    assert any(hit.id == stored.id for hit in hits)
