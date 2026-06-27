"""Domain constants."""
from mnemo.domain.memory_type import MemoryType

GLOBAL_PROJECT = "__global__"
DEFAULT_TYPE = MemoryType.WORKING_NOTES
# How many of the most query-relevant memories recall retrieves to ground its answer. Kept
# small: recall synthesizes a focused answer, not a digest of the whole project, and a large
# bundle only slows synthesis and dilutes the answer.
DEFAULT_RECALL_LIMIT = 15
# The agent-facing semantic search caps its page here: past ~20 hits the ranking is noise
# rather than signal, and the cap bounds the rerank over-fetch pool. browse (a filter list,
# e.g. "every decision") is intentionally not capped this tight.
SEARCH_LIMIT_MAX = 20
# A project's description is a short human- and (later) semantically-matched blurb — held to the
# same tight budget as a `rule` memory (MemoryType.RULE.max_tokens) so it stays a precise summary,
# not prose. Enforced in the create/update-project use cases (counting needs the token counter).
PROJECT_DESCRIPTION_MAX_TOKENS = 128
