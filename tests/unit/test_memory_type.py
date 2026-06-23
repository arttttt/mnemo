"""MemoryType is a proper enum (not a bare str), and each member carries its own token cap."""
from __future__ import annotations

from mnemo.domain.memory_type import MemoryType


def test_each_type_exposes_a_positive_token_cap():
    assert MemoryType.RULE.max_tokens == 128
    assert MemoryType.DECISION.max_tokens == 512
    assert all(isinstance(t.max_tokens, int) and t.max_tokens > 0 for t in MemoryType)


def test_rule_has_the_tightest_cap():
    others = (t for t in MemoryType if t is not MemoryType.RULE)
    assert MemoryType.RULE.max_tokens < min(t.max_tokens for t in others)


def test_value_roundtrips_and_the_type_is_not_a_str():
    assert MemoryType.RULE.value == "rule"
    assert MemoryType("rule") is MemoryType.RULE   # parse by value
    assert not isinstance(MemoryType.RULE, str)    # a distinct type, not a bare string
    assert MemoryType.RULE != "rule"               # no implicit string equality
