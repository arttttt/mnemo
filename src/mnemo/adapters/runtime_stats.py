"""Process memory reading for the model adapters' lifecycle logs.

``resource.getrusage`` gives peak RSS (monotonic — it never decreases), the honest
"how much did loading this model add" number on a CPU run. macOS reports bytes, Linux
kilobytes.
"""
from __future__ import annotations

import resource
import sys


def peak_rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024
