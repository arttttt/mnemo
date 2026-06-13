import asyncio

import pytest

pytest.importorskip("mcp")

from mnemo.adapters.mcp.server import build_mcp
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


def test_mcp_exposes_remember_and_search(tmp_path):
    mcp = build_mcp(_container(tmp_path))
    tools = asyncio.run(mcp.list_tools())
    names = {tool.name for tool in tools}
    assert {"remember", "search"} <= names
