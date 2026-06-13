import asyncio

import pytest

pytest.importorskip("mcp")

from mnemo.adapters.mcp.server import build_mcp
from mnemo.domain.memory_type import MemoryType
from mnemo.infrastructure.composition import build_container
from mnemo.infrastructure.config import Config


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


def test_mcp_exposes_the_agent_tools(tmp_path):
    assert {"remember", "search", "delete", "clear", "purge"} <= set(_tools(tmp_path))


def test_remember_advertises_allowed_types(tmp_path):
    schema = _tools(tmp_path)["remember"].inputSchema
    type_schema = schema["properties"]["type"]
    enum = type_schema.get("enum")
    assert enum is not None, type_schema
    assert set(enum) == {member.value for member in MemoryType}


def test_optional_params_expose_concrete_types(tmp_path):
    # Optional[...] would render as anyOf[T, null] (some MCP clients show that as
    # "unknown"); every parameter must expose a concrete type or enum instead.
    for name, tool in _tools(tmp_path).items():
        for pname, pschema in (tool.inputSchema.get("properties") or {}).items():
            assert "anyOf" not in pschema, f"{name}.{pname} uses anyOf"
            assert "type" in pschema or "enum" in pschema, f"{name}.{pname} lacks a type"


def test_only_required_params_are_marked_required(tmp_path):
    required = {
        name: tool.inputSchema.get("required", [])
        for name, tool in _tools(tmp_path).items()
    }
    assert required["remember"] == ["content"]
    assert required["search"] == ["query"]
    assert required["delete"] == ["ids"]
    assert required["clear"] == ["project"]
    assert required["purge"] == []
