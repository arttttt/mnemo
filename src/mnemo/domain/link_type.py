"""The kind of relationship a Link encodes between two memories."""
from enum import Enum


class LinkType(str, Enum):
    # Written automatically on a topic_key upsert: the successor supersedes the
    # prior record. More deterministic types (e.g. derived_from) and the
    # background worker's flag-only associative edges fit this same shape later.
    SUPERSEDES = "supersedes"
