"""Configuration (composition-root concern). Reads MNEMO_* environment variables."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

_Num = TypeVar("_Num", int, float)

DEFAULT_GENERATOR = "unsloth/gemma-4-E2B-it-qat-GGUF"
# Immutable Hugging Face commit selected and validated by the generator benchmark.
DEFAULT_GENERATOR_REVISION = "db01ae3ceeca98487bf3569814f832f5023cd48c"

# The default reranker: bge-reranker-v2-m3 (XLM-R cross-encoder), Q8 GGUF on llama.cpp/Metal —
# the LoCoMo + MIRACL A/B winner (bench/reranker-selection), built behind llmkit's GGUF reranker.
DEFAULT_RERANKER = "gpustack/bge-reranker-v2-m3-GGUF"
DEFAULT_RERANKER_FILE = "*Q8_0.gguf"  # GGUF glob in the repo
# Immutable Hugging Face commit so switching weights is a deliberate, pinned change.
DEFAULT_RERANKER_REVISION = "3093af03b1a635e67b084b1d8c03c5f5e020fd05"

def _int_env(name: str, default: str, *, minimum: int, maximum: int | None = None) -> int:
    """Parse a MNEMO_* integer at the config boundary, failing fast with a named, range-checked
    error — so a typo or out-of-range value surfaces at startup instead of an opaque later crash
    (a raw traceback in the CLI, or the service's "exited before listening")."""
    raw = os.environ.get(name, default)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    return _in_range(name, value, minimum, maximum, exclusive_min=False)


def _float_env(
    name: str, default: str, *, minimum: float, maximum: float | None = None,
    exclusive_min: bool = False,
) -> float:
    """Parse a MNEMO_* float at the config boundary, with the same fail-fast, named, range-checked
    contract as _int_env. exclusive_min rejects the bound itself (e.g. an idle-check interval must
    be strictly > 0 — a 0 would busy-loop)."""
    raw = os.environ.get(name, default)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not math.isfinite(value):  # nan/inf parse but slip every range check (nan compares False)
        raise ValueError(f"{name} must be a finite number, got {raw!r}")
    return _in_range(name, value, minimum, maximum, exclusive_min=exclusive_min)


def _in_range(
    name: str, value: _Num, minimum: float, maximum: float | None, *, exclusive_min: bool
) -> _Num:
    too_low = value <= minimum if exclusive_min else value < minimum
    if too_low:
        rel = ">" if exclusive_min else ">="
        raise ValueError(f"{name} must be {rel} {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


@dataclass(frozen=True)
class Config:
    data_dir: str
    embedder: str
    models_dir: str = ""            # where local models are cached (default ~/.mnemo/models)
    embed_max_tokens: int = 2048    # the embedder's operational window cap (pplx)
    sqlite_path: str = ""
    host: str = "127.0.0.1"   # the shared service binds localhost-only
    port: int = 8765
    idle_grace_seconds: float = 300.0         # exit this long after the last connector leaves
    idle_check_interval_seconds: float = 5.0  # how often the service sweeps for live connectors
    service_ready_timeout: float = 120.0      # how long a spawn waits for the service to listen
    #                                           (covers a cold model download+load)
    # Deferred embedding (the service's async worker pool; docs/03-architecture.md).
    embed_workers: int = 1                     # worker threads = embedder pool size (N independent instances) = max parallel encodes; the RAM knob (default 1 = safe)
    embed_queue_max: int = 256                 # backlog cap; above it a write embeds synchronously
    embed_max_retries: int = 3                 # retries before a memory is left lexical-only
    embed_drain_timeout: float = 30.0          # how long idle-exit waits for the queue to drain
    # Recall pipeline models (benchmark-selected). Set any to "off" to drop that stage.
    # Reranker: bge-reranker-v2-m3 Q8 GGUF on llama.cpp/Metal (the A/B winner) by default, loaded
    # Transient (gated, load-on-call) — see composition._build_reranker. "off" drops the stage.
    reranker: str = DEFAULT_RERANKER                              # MNEMO_RERANKER: GGUF/ONNX cross-encoder repo / path / "off"
    reranker_file: str = DEFAULT_RERANKER_FILE                    # MNEMO_RERANKER_FILE: GGUF glob in the repo
    reranker_revision: str | None = None                          # MNEMO_RERANKER_REVISION: defaults to the pin for DEFAULT_RERANKER
    # Generator: Gemma 4 E2B-it, official QAT GGUF (near-lossless Q4) — best faithful synthesis
    # per the bench, at the lightest RAM; driven through its chat template (see _build_generator).
    generator: str = DEFAULT_GENERATOR                            # MNEMO_GENERATOR: HF GGUF repo / path / "off"
    generator_revision: str | None = None                         # MNEMO_GENERATOR_REVISION: defaults to the pin for DEFAULT_GENERATOR
    generator_file: str = "*UD-Q4_K_XL.gguf"                     # MNEMO_GENERATOR_FILE: GGUF glob in the repo
    generator_context: int = 65536                              # MNEMO_GENERATOR_CONTEXT: n_ctx window (holds the recall bundle + answer); KV-cache RAM knob
    rerank_top_k: int = 20                                       # how many candidates the reranker keeps
    generator_max_tokens: int = 2048                            # synthesis output token cap

    @staticmethod
    def from_env() -> "Config":
        data_dir = os.path.expanduser(os.environ.get("MNEMO_DATA_DIR", "~/.mnemo/data"))
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        return Config(
            data_dir=data_dir,
            embedder=os.environ.get("MNEMO_EMBEDDER", "pplx"),
            models_dir=os.path.expanduser(
                os.environ.get("MNEMO_MODELS_DIR", "~/.mnemo/models")
            ),
            embed_max_tokens=_int_env("MNEMO_EMBED_MAX_TOKENS", "2048", minimum=1),
            sqlite_path=os.environ.get(
                "MNEMO_SQLITE_PATH", os.path.join(data_dir, "memory.db")
            ),
            host=os.environ.get("MNEMO_HOST", "127.0.0.1"),
            port=_int_env("MNEMO_PORT", "8765", minimum=1, maximum=65535),
            idle_grace_seconds=_float_env("MNEMO_IDLE_GRACE_SECONDS", "300", minimum=0),
            idle_check_interval_seconds=_float_env(
                "MNEMO_IDLE_CHECK_INTERVAL_SECONDS", "5", minimum=0, exclusive_min=True
            ),
            service_ready_timeout=_float_env(
                "MNEMO_SERVICE_READY_TIMEOUT", "120", minimum=0, exclusive_min=True
            ),
            embed_workers=_int_env("MNEMO_EMBED_WORKERS", "1", minimum=1),
            embed_queue_max=_int_env("MNEMO_EMBED_QUEUE_MAX", "256", minimum=1),
            embed_max_retries=_int_env("MNEMO_EMBED_MAX_RETRIES", "3", minimum=0),
            embed_drain_timeout=_float_env("MNEMO_EMBED_DRAIN_TIMEOUT", "30", minimum=0),
            reranker=os.environ.get("MNEMO_RERANKER", DEFAULT_RERANKER),
            reranker_file=os.environ.get("MNEMO_RERANKER_FILE", DEFAULT_RERANKER_FILE),
            reranker_revision=os.environ.get("MNEMO_RERANKER_REVISION") or None,
            generator=os.environ.get("MNEMO_GENERATOR", DEFAULT_GENERATOR),
            generator_revision=os.environ.get("MNEMO_GENERATOR_REVISION") or None,
            generator_file=os.environ.get("MNEMO_GENERATOR_FILE", "*UD-Q4_K_XL.gguf"),
            generator_context=_int_env("MNEMO_GENERATOR_CONTEXT", "65536", minimum=1),
            rerank_top_k=_int_env("MNEMO_RERANK_TOP_K", "20", minimum=1),
            generator_max_tokens=_int_env("MNEMO_GENERATOR_MAX_TOKENS", "2048", minimum=1),
        )
