import json

import pytest

testing = pytest.importorskip("typer.testing")


def _runner_and_app(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMO_EMBEDDER", "hash")
    monkeypatch.setenv("MNEMO_STORE", "memory")
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MNEMO_STORE_PATH", str(tmp_path / "memory.json"))
    from mnemo.adapters.cli.app import app

    return testing.CliRunner(), app


def test_cli_store_then_search(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)

    stored = runner.invoke(
        app, ["store", "JWT refresh rotation", "--type", "decision", "--project", "api"]
    )
    assert stored.exit_code == 0, stored.output
    memory_id = json.loads(stored.stdout)["id"]

    found = runner.invoke(app, ["search", "jwt rotation", "--project", "api"])
    assert found.exit_code == 0, found.output
    assert memory_id in found.stdout
