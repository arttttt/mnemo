"""Port: something that reports how many connectors are currently alive."""
from __future__ import annotations

from typing import Protocol


class LivenessProbe(Protocol):
    def live_count(self) -> int: ...
