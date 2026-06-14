"""The `mnemo setup` command end to end through the real CLI app.

CLI-client detection is neutralized (which -> None) so these never invoke a real
client binary that happens to be installed; only the file-based clients (whose
config dirs are created to simulate "installed") are wired, against a temp home.
"""
import json
import shutil

from typer.testing import CliRunner

from mnemo.adapters.cli.app import app


def test_setup_wires_a_named_file_client(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    (tmp_path / ".cursor").mkdir()
    result = CliRunner().invoke(app, ["setup", "cursor"], env={"HOME": str(tmp_path)})
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / ".cursor" / "mcp.json").read_text())
    assert "mnemo" in data["mcpServers"]


def test_setup_detects_lists_and_wires_the_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)  # no CLI clients on PATH
    (tmp_path / ".cursor").mkdir()
    (tmp_path / ".config" / "opencode").mkdir(parents=True)

    result = CliRunner().invoke(app, ["setup"], env={"HOME": str(tmp_path)}, input="all\n")
    assert result.exit_code == 0, result.output
    assert "cursor" in result.output and "opencode" in result.output
    assert (tmp_path / ".cursor" / "mcp.json").exists()
    assert (tmp_path / ".config" / "opencode" / "opencode.json").exists()


def test_setup_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    (tmp_path / ".cursor").mkdir()
    result = CliRunner().invoke(app, ["setup", "--dry-run"], env={"HOME": str(tmp_path)})
    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".cursor" / "mcp.json").exists()


def test_setup_rejects_unknown_client(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    result = CliRunner().invoke(app, ["setup", "nope"], env={"HOME": str(tmp_path)})
    assert result.exit_code != 0
