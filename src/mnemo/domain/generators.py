"""Id and timestamp generators."""
import secrets
from datetime import datetime, timezone


def new_id() -> str:
    # 160-bit cryptographically-random id (40 hex chars) — longer than a uuid4.
    return secrets.token_hex(20)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()
