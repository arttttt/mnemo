"""Result of migrating memories between stores."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MigrationResult:
    source_total: int   # memories found in the source store
    added: int          # written to the target this run
    skipped: int        # already present in the target (idempotent re-run)
