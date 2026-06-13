"""Text normalization and content fingerprinting for exact dedup."""
import hashlib
import re


def normalize(text: str) -> str:
    """Canonical form for hashing/dedup: collapsed whitespace, lowercased."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def content_hash(text: str) -> str:
    """Stable content fingerprint used for exact dedup."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()
