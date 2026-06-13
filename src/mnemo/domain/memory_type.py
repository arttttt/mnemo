"""The kind of a memory — shapes how it is stored and retrieved."""
from enum import Enum


class MemoryType(str, Enum):
    DECISION = "decision"
    DEBUG = "debug"
    PROGRESS = "progress"
    FEATURE = "feature"
    RESEARCH = "research"
    CODE_SNIPPET = "code-snippet"
    RULE = "rule"
    LEARNING = "learning"
    DISCUSSION = "discussion"
    DESIGN = "design"
    WORKING_NOTES = "working-notes"
