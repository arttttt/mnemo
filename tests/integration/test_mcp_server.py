import asyncio

import pytest

pytest.importorskip("mcp")

from mnemo.adapters.mcp.server import build_mcp
from mnemo.domain.memory import MemoryType
from mnemo.infrastructure.config import Config
from mnemo.infrastructure.container import build_container


def _container(tmp_path):
    config = Config(
        data_dir=str(tmp_path),
        embedder="hash",
        store="memory",
        store_path=str(tmp_path / "memory.json"),
    )
    return build_container(config)


def _tools(tmp_path):
    mcp = build_mcp(_container(tmp_path))
    return {tool.name: tool for tool in asyncio.run(mcp.list_tools())}


def test_mcp_exposes_remember_and_search(tmp_path):
    assert {"remember", "search"} <= set(_tools(tmp_path))


def test_remember_advertises_allowed_types(tmp_path):
    schema = _tools(tmp_path)["remember"].inputSchema
    type_schema = schema["properties"]["type"]
    enum = type_schema.get("enum")
    assert enum is not None, type_schema
    assert set(enum) == {member.value for member in MemoryType}
