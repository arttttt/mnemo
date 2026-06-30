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


def test_setup_named_client_json_reports_the_install_result(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    (tmp_path / ".cursor").mkdir()
    result = CliRunner().invoke(app, ["setup", "cursor", "--json"], env={"HOME": str(tmp_path)})
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload == [
        {"client": "cursor", "status": "ok",
         "target": str(tmp_path / ".cursor" / "mcp.json"), "message": ""}
    ]
    assert (tmp_path / ".cursor" / "mcp.json").exists()  # actually wired


def test_setup_detection_json_is_non_interactive_and_reports_only(tmp_path, monkeypatch):
    # With --json and no client/--all, setup must NOT prompt (an agent can't answer): it
    # reports the detected clients as JSON and writes nothing.
    monkeypatch.setattr(shutil, "which", lambda name: None)
    (tmp_path / ".cursor").mkdir()
    (tmp_path / ".config" / "opencode").mkdir(parents=True)

    result = CliRunner().invoke(app, ["setup", "--json"], env={"HOME": str(tmp_path)})
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert {entry["client"] for entry in payload} == {"cursor", "opencode"}
    assert all(entry["detected"] is True for entry in payload)
    assert not (tmp_path / ".cursor" / "mcp.json").exists()  # reporting only — nothing written


def test_setup_all_json_wires_every_detected_client(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    (tmp_path / ".cursor").mkdir()
    (tmp_path / ".config" / "opencode").mkdir(parents=True)

    result = CliRunner().invoke(app, ["setup", "--all", "--json"], env={"HOME": str(tmp_path)})
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert {entry["client"] for entry in payload} == {"cursor", "opencode"}
    assert all(entry["status"] == "ok" for entry in payload)
    assert (tmp_path / ".cursor" / "mcp.json").exists()
    assert (tmp_path / ".config" / "opencode" / "opencode.json").exists()


def test_setup_dry_run_json_emits_a_plan_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    (tmp_path / ".cursor").mkdir()
    result = CliRunner().invoke(
        app, ["setup", "cursor", "--dry-run", "--json"], env={"HOME": str(tmp_path)}
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    assert plan[0]["client"] == "cursor" and "action" in plan[0]
    assert not (tmp_path / ".cursor" / "mcp.json").exists()
