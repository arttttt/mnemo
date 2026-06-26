"""The composition root wires a working use case from config, on each backend."""
import pytest

from mnemo.infrastructure.composition import (
    _build_generator,
    _build_reranker,
    build_container,
)
from mnemo.infrastructure.config import (
    DEFAULT_GENERATOR_REVISION,
    DEFAULT_RERANKER,
    DEFAULT_RERANKER_FILE,
    DEFAULT_RERANKER_REVISION,
    Config,
)


def test_from_env_defaults(monkeypatch, tmp_path):
    for var in ("MNEMO_SQLITE_PATH", "MNEMO_EMBEDDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))

    config = Config.from_env()
    assert config.embedder == "pplx"  # the default embedder
    assert config.sqlite_path == str(tmp_path / "memory.db")


def test_from_env_reads_model_revisions(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MNEMO_RERANKER_REVISION", "reranker-commit")
    monkeypatch.setenv("MNEMO_GENERATOR_REVISION", "generator-commit")

    config = Config.from_env()

    assert config.reranker_revision == "reranker-commit"
    assert config.generator_revision == "generator-commit"


def test_default_generator_uses_the_pinned_revision(monkeypatch):
    captured = {}

    def capture(config):
        captured["config"] = config
        return object()

    monkeypatch.setattr("llmkit.build.build_generator", capture)

    _build_generator(Config(data_dir="/tmp", embedder="hash"))

    assert captured["config"].source.revision == DEFAULT_GENERATOR_REVISION


def test_custom_hf_generator_requires_a_revision():
    config = Config(data_dir="/tmp", embedder="hash", generator="org/custom-gguf")

    with pytest.raises(ValueError, match="MNEMO_GENERATOR_REVISION"):
        _build_generator(config)


def test_local_generator_does_not_need_a_revision(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.touch()
    captured = {}

    def capture(config):
        captured["config"] = config
        return object()

    monkeypatch.setattr("llmkit.build.build_generator", capture)

    _build_generator(Config(data_dir="/tmp", embedder="hash", generator=str(model)))

    assert captured["config"].source.revision is None


def test_custom_hf_reranker_requires_and_forwards_a_revision(monkeypatch):
    without_revision = Config(data_dir="/tmp", embedder="hash", reranker="org/reranker")
    with pytest.raises(ValueError, match="MNEMO_RERANKER_REVISION"):
        _build_reranker(without_revision)

    captured = {}

    def capture(config):
        captured["config"] = config
        return object()

    monkeypatch.setattr("llmkit.build.build_reranker", capture)
    _build_reranker(
        Config(
            data_dir="/tmp",
            embedder="hash",
            reranker="org/reranker",
            reranker_revision="immutable-commit",
        )
    )

    assert captured["config"].source.revision == "immutable-commit"


def test_default_reranker_wires_the_bge_gguf_with_its_pinned_revision(monkeypatch):
    # The default config builds the bge GGUF cross-encoder on llama.cpp, Transient (gated),
    # with the pinned revision — no model is loaded (build is lazy; here build_reranker is patched).
    captured = {}

    def capture(config):
        captured["config"] = config
        return object()

    monkeypatch.setattr("llmkit.build.build_reranker", capture)

    _build_reranker(Config(data_dir="/tmp", embedder="hash"))  # default reranker = DEFAULT_RERANKER

    source = captured["config"].source
    assert source.model == DEFAULT_RERANKER
    assert source.filename == DEFAULT_RERANKER_FILE
    assert source.revision == DEFAULT_RERANKER_REVISION
    assert source.context_tokens == 1024


def test_reranker_off_builds_nothing():
    assert _build_reranker(Config(data_dir="/tmp", embedder="hash", reranker="off")) is None


def test_local_gguf_reranker_does_not_need_a_revision(tmp_path, monkeypatch):
    model = tmp_path / "reranker.gguf"
    model.touch()
    captured = {}

    def capture(config):
        captured["config"] = config
        return object()

    monkeypatch.setattr("llmkit.build.build_reranker", capture)

    _build_reranker(Config(data_dir="/tmp", embedder="hash", reranker=str(model)))

    assert captured["config"].source.revision is None


def test_build_container_wires_the_sqlite_backend(tmp_path):
    pytest.importorskip("sqlite_vec")
    from mnemo.adapters.store.sqlite_vec_repository import SqliteRepositoryImpl

    config = Config(
        data_dir=str(tmp_path),
        embedder="hash",
        reranker="off",  # keep the wiring test offline: the default reranker is a GGUF download
        sqlite_path=str(tmp_path / "memory.db"),
    )
    container = build_container(config)
    assert isinstance(container.repository, SqliteRepositoryImpl)

    container.create_project.execute("api")
    stored = container.remember.execute(content="wired on sqlite", project="api")
    hits = container.search.execute(query="wired on sqlite", project="api")
    assert any(hit.id == stored.id for hit in hits)
