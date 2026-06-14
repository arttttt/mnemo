"""connector_command resolves an absolute launch argv for the connector."""
import shutil

from mnemo.adapters.setup.connector_command import connector_command


def test_uses_absolute_path_when_on_path(monkeypatch):
    monkeypatch.setattr(
        shutil, "which", lambda name: "/abs/bin/mnemo-mcp" if name == "mnemo-mcp" else None
    )
    assert connector_command() == ["/abs/bin/mnemo-mcp"]


def test_falls_back_to_uv_run_when_absent(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/abs/bin/uv" if name == "uv" else None)
    argv = connector_command()
    assert argv[0] == "/abs/bin/uv"
    assert argv[1:3] == ["run", "--directory"]
    assert argv[-1] == "mnemo-mcp"
