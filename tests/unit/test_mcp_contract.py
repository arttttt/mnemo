"""The MCP boundary enums must stay in sync with the domain enums (DRY guard)."""
from typing import get_args

from mnemo.adapters.mcp.server import MemoryTypeName, SearchScope, StoreScope
from mnemo.domain.memory import MemoryType, Scope


def test_mcp_memory_type_literal_matches_domain():
    assert set(get_args(MemoryTypeName)) == {member.value for member in MemoryType}


def test_mcp_store_scope_literal_matches_domain():
    assert set(get_args(StoreScope)) == {member.value for member in Scope}


def test_mcp_search_scope_extends_store_scope_with_all():
    assert set(get_args(SearchScope)) == {member.value for member in Scope} | {"all"}
