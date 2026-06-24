"""Process memory readings for the runtimes' lifecycle logs.

Two complementary, dependency-free numbers:

* ``peak_rss_mb`` — peak RSS via ``resource.getrusage`` (monotonic high-water; never
  decreases). Good for "how much did loading this model add", but it cannot show a release.
* ``current_rss_mb`` — live resident set size *right now*; it drops when memory is freed, so
  an unload visibly falls. ``/proc/self/statm`` on Linux, ``ps`` on macOS/BSD; falls back to
  the peak if neither is available.

Unit notes: macOS ``ru_maxrss`` is bytes, Linux kilobytes; ``ps -o rss`` and ``statm``
(pages × page size) are kibibytes.
"""
from __future__ import annotations

import os
import resource
import subprocess
import sys


def peak_rss_mb() -> float:
    """Peak (high-water) RSS in MiB. Monotonic — never decreases, so it cannot show a free."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def current_rss_mb() -> float:
    """Live resident set size in MiB — unlike the peak, this drops when memory is freed.

    Dependency-free per OS and it never raises: a logging helper must not break its caller,
    so any failure falls back to ``peak_rss_mb``.
    """
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/self/statm") as statm:
                resident_pages = int(statm.read().split()[1])
            return resident_pages * resource.getpagesize() / (1024 * 1024)
        # macOS/BSD have no /proc; `ps` reports RSS in kibibytes.
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True, text=True, check=True,
        )
        return int(out.stdout.strip()) / 1024
    except Exception:  # noqa: BLE001 — never let a memory probe break a log call
        return peak_rss_mb()
