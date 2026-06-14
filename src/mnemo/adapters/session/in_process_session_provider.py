"""In-process session provider: one lazily-generated id per run."""
from __future__ import annotations

from mnemo.domain.generators import new_id


class InProcessSessionProvider:
    """One session id per run. It is generated on the first call and returned for
    the rest of the run; a read-only run never triggers generation. One
    process/connection = one run today; a shared-process deployment swaps this
    for a per-connection provider behind the same port.
    """

    def __init__(self) -> None:
        self._session_id: str | None = None

    def current_session_id(self) -> str:
        if self._session_id is None:
            self._session_id = new_id()
        return self._session_id
