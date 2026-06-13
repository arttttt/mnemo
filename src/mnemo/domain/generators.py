"""Id and timestamp generators."""
import secrets
from datetime import datetime, timedelta, timezone


def new_id() -> str:
    # 160-bit cryptographically-random id (40 hex chars) — longer than a uuid4.
    return secrets.token_hex(20)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iso_days_ago(days: int) -> str:
    # Same UTC ISO format as now(), so string comparison stays chronological.
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
