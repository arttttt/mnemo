"""Config.from_env validation — every numeric MNEMO_* fails fast with a named, range-checked error."""
import pytest

from mnemo.infrastructure.config import (
    DEFAULT_RERANKER,
    DEFAULT_RERANKER_FILE,
    Config,
)

# Every numeric MNEMO_* the fixture must clear so a polluted shell can't skew the defaults.
NUMERIC_VARS = [
    "MNEMO_PORT",
    "MNEMO_EMBED_MAX_TOKENS",
    "MNEMO_EMBED_WORKERS",
    "MNEMO_EMBED_QUEUE_MAX",
    "MNEMO_EMBED_MAX_RETRIES",
    "MNEMO_RERANK_TOP_K",
    "MNEMO_GENERATOR_MAX_TOKENS",
    "MNEMO_IDLE_GRACE_SECONDS",
    "MNEMO_IDLE_CHECK_INTERVAL_SECONDS",
    "MNEMO_SERVICE_READY_TIMEOUT",
    "MNEMO_EMBED_DRAIN_TIMEOUT",
]

# (var, a value that must be rejected) — non-numeric and out-of-range, one+ per var.
REJECTED = [
    ("MNEMO_PORT", "notanint"),                      # not an integer
    ("MNEMO_PORT", "0"),                             # < 1
    ("MNEMO_PORT", "99999"),                         # > 65535
    ("MNEMO_EMBED_MAX_TOKENS", "0"),                 # < 1
    ("MNEMO_EMBED_MAX_TOKENS", "-1"),
    ("MNEMO_EMBED_WORKERS", "0"),                    # < 1
    ("MNEMO_EMBED_WORKERS", "lots"),                 # not an integer
    ("MNEMO_EMBED_QUEUE_MAX", "-5"),                 # < 1
    ("MNEMO_EMBED_MAX_RETRIES", "-1"),               # < 0
    ("MNEMO_RERANK_TOP_K", "0"),                     # < 1
    ("MNEMO_GENERATOR_MAX_TOKENS", "0"),             # < 1
    ("MNEMO_IDLE_GRACE_SECONDS", "-1"),              # < 0
    ("MNEMO_IDLE_GRACE_SECONDS", "soon"),            # not a number
    ("MNEMO_IDLE_CHECK_INTERVAL_SECONDS", "0"),      # must be strictly > 0 (0 would busy-loop)
    ("MNEMO_IDLE_CHECK_INTERVAL_SECONDS", "-1"),
    ("MNEMO_SERVICE_READY_TIMEOUT", "0"),            # must be strictly > 0
    ("MNEMO_SERVICE_READY_TIMEOUT", "inf"),          # parses as float but isn't finite
    ("MNEMO_EMBED_DRAIN_TIMEOUT", "-0.5"),           # < 0
    ("MNEMO_EMBED_DRAIN_TIMEOUT", "nan"),            # nan would disable the timeout cap
    ("MNEMO_IDLE_GRACE_SECONDS", "inf"),             # inf grace → the service never idle-exits
    ("MNEMO_IDLE_CHECK_INTERVAL_SECONDS", "nan"),    # nan interval → busy-loop
]

# (var, value, attr, expected) — boundary values that must be ACCEPTED.
ACCEPTED = [
    ("MNEMO_PORT", "1", "port", 1),
    ("MNEMO_PORT", "65535", "port", 65535),
    ("MNEMO_EMBED_WORKERS", "4", "embed_workers", 4),
    ("MNEMO_EMBED_MAX_RETRIES", "0", "embed_max_retries", 0),          # 0 = no retries, valid
    ("MNEMO_IDLE_GRACE_SECONDS", "0", "idle_grace_seconds", 0.0),      # 0 = exit immediately, valid
    ("MNEMO_EMBED_DRAIN_TIMEOUT", "0", "embed_drain_timeout", 0.0),    # 0 = don't wait, valid
    ("MNEMO_IDLE_CHECK_INTERVAL_SECONDS", "0.5", "idle_check_interval_seconds", 0.5),
]


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))  # keep from_env off the real ~/.mnemo
    for var in NUMERIC_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_defaults_are_valid(env):
    config = Config.from_env()
    assert config.port == 8765
    assert config.embed_workers == 1
    assert config.embed_max_retries == 3
    assert config.idle_check_interval_seconds == 5.0


def test_reranker_defaults_to_the_bge_gguf():
    # The dataclass default (independent of the environment) is the bge GGUF cross-encoder.
    config = Config(data_dir="/tmp", embedder="hash")
    assert config.reranker == DEFAULT_RERANKER
    assert config.reranker_file == DEFAULT_RERANKER_FILE


@pytest.mark.parametrize("name,value", REJECTED)
def test_invalid_numeric_env_is_rejected_with_a_named_error(env, name, value):
    env.setenv(name, value)
    with pytest.raises(ValueError, match=name):  # the message names the offending var
        Config.from_env()


@pytest.mark.parametrize("name,value,attr,expected", ACCEPTED)
def test_valid_boundary_values_are_accepted(env, name, value, attr, expected):
    env.setenv(name, value)
    assert getattr(Config.from_env(), attr) == expected
