"""Domain constants."""
from mnemo.domain.memory_type import MemoryType

GLOBAL_PROJECT = "__global__"
DEFAULT_TYPE = MemoryType.WORKING_NOTES
# A memory is one focused unit: cap its length so large blobs aren't stored (they bloat the
# store and dilute retrieval). This is the policy cap; the embedder's window is a separate,
# usually larger hard ceiling — the effective limit is the stricter of the two.
DEFAULT_MAX_MEMORY_TOKENS = 512
# How many of the most query-relevant memories recall retrieves to ground its answer. Kept
# small: recall synthesizes a focused answer, not a digest of the whole project, and a large
# bundle only slows synthesis and dilutes the answer.
DEFAULT_RECALL_LIMIT = 15
