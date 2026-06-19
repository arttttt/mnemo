"""llmkit is a standalone inference package — it must never depend on mnemo."""
from __future__ import annotations

from pathlib import Path


def test_llmkit_never_imports_mnemo():
    root = Path(__file__).resolve().parents[2] / "src" / "llmkit"
    offenders = []
    for path in root.rglob("*.py"):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith(("import mnemo", "from mnemo")):
                offenders.append(f"{path.relative_to(root)}:{number}: {stripped}")
    assert not offenders, "llmkit must not import mnemo:\n" + "\n".join(offenders)
