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


def test_cli_delete_clear_purge_and_stats(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)

    one = json.loads(runner.invoke(app, ["store", "one", "--project", "api"]).stdout)["id"]
    runner.invoke(app, ["store", "two", "--project", "api"])
    runner.invoke(app, ["store", "three", "--project", "other"])

    assert json.loads(runner.invoke(app, ["stats"]).stdout)["total"] == 3
    assert json.loads(runner.invoke(app, ["delete", one]).stdout)["deleted"] == 1
    assert json.loads(runner.invoke(app, ["clear", "api"]).stdout)["deleted"] == 1
    assert json.loads(runner.invoke(app, ["purge"]).stdout)["deleted"] == 1
    assert json.loads(runner.invoke(app, ["stats"]).stdout)["total"] == 0


@pytest.mark.heavy
def test_cli_migrate_json_store_into_lancedb(tmp_path, monkeypatch):
    pytest.importorskip("lancedb")
    monkeypatch.setenv("MNEMO_EMBEDDER", "hash")
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MNEMO_STORE_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setenv("MNEMO_LANCEDB_URI", str(tmp_path / "memory"))
    from mnemo.adapters.cli.app import app

    runner = testing.CliRunner()

    monkeypatch.setenv("MNEMO_STORE", "memory")  # seed the JSON store
    runner.invoke(app, ["store", "jwt rotation", "--type", "decision", "--project", "api"])

    migrated = json.loads(runner.invoke(app, ["migrate"]).stdout)
    assert migrated["source_total"] == 1 and migrated["added"] == 1

    monkeypatch.setenv("MNEMO_STORE", "lancedb")  # read it back from LanceDB
    found = runner.invoke(app, ["search", "jwt rotation", "--project", "api"])
    assert found.exit_code == 0, found.output
    assert "jwt rotation" in found.stdout
