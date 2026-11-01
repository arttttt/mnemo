"""Replace a client's config file without ever leaving it half-written.

The new content goes to a temporary file next to the real one, which then takes its
place in one rename, so a crash or a full disk leaves either the old file or the new
one, never a truncated settings file. A config file that is a symlink (into a dotfiles
checkout, say) stays one: the file it points to is replaced. An existing file keeps
its permission bits; a new one is readable by its owner only.
"""
from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path


def replace_file(path: Path, data: bytes) -> None:
    target = Path(os.path.realpath(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = target.stat().st_mode & 0o7777 if target.exists() else None
    handle, temp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        if mode is not None:
            os.chmod(temp, mode)
        os.replace(temp, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp)
        raise
