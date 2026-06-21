"""Domain constants."""
from mnemo.domain.memory_type import MemoryType

GLOBAL_PROJECT = "__global__"
DEFAULT_TYPE = MemoryType.WORKING_NOTES
# A memory is one focused unit: cap its length so large blobs aren't stored (they bloat the
# store and dilute retrieval). This is the policy cap; the embedder's window is a separate,
# usually larger hard ceiling — the effective limit is the stricter of the two.
DEFAULT_MAX_MEMORY_TOKENS = 512
