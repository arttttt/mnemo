"""The scope a memory belongs to."""
from enum import Enum


class Scope(str, Enum):
    PROJECT = "project"
    GLOBAL = "global"
