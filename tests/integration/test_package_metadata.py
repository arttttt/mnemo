"""Installed distribution metadata exposes a complete default runtime."""
from __future__ import annotations

import re
from importlib.metadata import metadata, requires


def _package_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    assert match is not None
    return match.group(0).lower().replace("_", "-")


def test_default_model_runtimes_are_required_dependencies():
    declared = requires("mnemo") or []
    required = {
        _package_name(requirement)
        for requirement in declared
        if "extra ==" not in requirement
    }

    assert {
        "onnxruntime",
        "tokenizers",
        "numpy",
        "huggingface-hub",
        "llama-cpp-python",
    } <= required


def test_removed_default_runtime_extras_are_not_advertised():
    extras = set(metadata("mnemo").get_all("Provides-Extra") or [])

    assert "pplx" not in extras
    assert "recall" not in extras
    assert "embed" not in extras
