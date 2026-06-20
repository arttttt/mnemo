"""Config.from_env validation — MNEMO_EMBED_WORKERS must be a positive integer."""
import pytest

from mnemo.infrastructure.config import Config


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))  # keep from_env off the real ~/.mnemo
    monkeypatch.delenv("MNEMO_EMBED_WORKERS", raising=False)
    return monkeypatch


def test_embed_workers_defaults_to_one(env):
    assert Config.from_env().embed_workers == 1


def test_embed_workers_reads_a_valid_value(env):
    env.setenv("MNEMO_EMBED_WORKERS", "4")
    assert Config.from_env().embed_workers == 4


def test_embed_workers_zero_or_negative_is_rejected(env):
    for bad in ("0", "-2"):
        env.setenv("MNEMO_EMBED_WORKERS", bad)
        with pytest.raises(ValueError, match="MNEMO_EMBED_WORKERS"):
            Config.from_env()


def test_embed_workers_non_integer_is_rejected(env):
    env.setenv("MNEMO_EMBED_WORKERS", "lots")
    with pytest.raises(ValueError, match="MNEMO_EMBED_WORKERS"):
        Config.from_env()
