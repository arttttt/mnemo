import json

import pytest

testing = pytest.importorskip("typer.testing")


def _runner_and_app(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMO_EMBEDDER", "hash")
    monkeypatch.setenv("MNEMO_STORE", "memory")
    monkeypatch.setenv("MNEMO_RERANKER", "off")    # keep tests offline: no model download
    monkeypatch.setenv("MNEMO_GENERATOR", "off")
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


def test_cli_search_project_scope_without_project_fails_cleanly(tmp_path, monkeypatch):
    # --scope defaults to 'project'; with no --project there is nothing to scope to,
    # so the command exits non-zero with an actionable message, not a stack trace.
    runner, app = _runner_and_app(tmp_path, monkeypatch)

    result = runner.invoke(app, ["search", "anything"])
    assert result.exit_code != 0
    assert "project" in result.output
    assert "Traceback" not in result.output


def test_cli_search_global_scope_needs_no_project(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)

    result = runner.invoke(app, ["search", "anything", "--scope", "global"])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "[]"  # empty store, but the command runs end-to-end


def test_cli_browse_lists_memories_without_a_query(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    a = json.loads(runner.invoke(app, ["store", "alpha", "--project", "api"]).stdout)["id"]
    b = json.loads(runner.invoke(app, ["store", "beta", "--project", "api"]).stdout)["id"]

    result = runner.invoke(app, ["browse", "--project", "api"])
    assert result.exit_code == 0, result.output
    hits = json.loads(result.stdout)
    created = [hit["created_at"] for hit in hits]
    assert created == sorted(created, reverse=True)  # newest first
    assert {hit["id"] for hit in hits} == {a, b}
    assert all("score" not in hit for hit in hits)  # browse carries no score


def test_cli_browse_project_scope_without_project_fails_cleanly(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    result = runner.invoke(app, ["browse"])  # --scope defaults to 'project', no --project
    assert result.exit_code != 0
    assert "project" in result.output
    assert "Traceback" not in result.output


def test_cli_store_sets_tags_and_files(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)

    stored = runner.invoke(
        app,
        ["store", "jwt rotation", "--project", "api",
         "--tag", "auth", "--file", "src/auth.py"],
    )
    assert stored.exit_code == 0, stored.output
    memory_id = json.loads(stored.stdout)["id"]

    # The metadata was actually set: search filters on tag/file find it...
    by_tag = runner.invoke(app, ["search", "jwt rotation", "--project", "api", "--tag", "auth"])
    assert memory_id in by_tag.stdout
    by_file = runner.invoke(
        app, ["search", "jwt rotation", "--project", "api", "--file", "src/auth.py"]
    )
    assert memory_id in by_file.stdout

    # ...and a non-matching tag filter excludes it (proves it wasn't silently dropped).
    miss = runner.invoke(app, ["search", "jwt rotation", "--project", "api", "--tag", "cache"])
    assert memory_id not in miss.stdout


def test_cli_delete_clear_purge_and_stats(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)

    one = json.loads(runner.invoke(app, ["store", "one", "--project", "api"]).stdout)["id"]
    runner.invoke(app, ["store", "two", "--project", "api"])
    runner.invoke(app, ["store", "three", "--project", "other"])

    stats = json.loads(runner.invoke(app, ["stats"]).stdout)
    assert stats["total"] == 3
    assert stats["pending"] == 0  # the CLI embeds inline (sync scheduler), so nothing pending
    assert json.loads(runner.invoke(app, ["delete", one]).stdout)["deleted"] == 1
    assert json.loads(runner.invoke(app, ["clear", "api"]).stdout)["deleted"] == 1
    assert json.loads(runner.invoke(app, ["purge"]).stdout)["deleted"] == 1
    assert json.loads(runner.invoke(app, ["stats"]).stdout)["total"] == 0


def test_cli_clear_scope_global_targets_globals(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    runner.invoke(app, ["store", "proj note", "--project", "api"])
    runner.invoke(app, ["store", "global rule", "--scope", "global", "--type", "rule"])

    cleared = runner.invoke(app, ["clear", "--scope", "global"])
    assert cleared.exit_code == 0, cleared.output
    assert json.loads(cleared.stdout)["deleted"] == 1
    assert json.loads(runner.invoke(app, ["stats"]).stdout)["total"] == 1  # project note survives


def test_cli_clear_project_scope_without_project_fails_cleanly(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    result = runner.invoke(app, ["clear"])  # --scope defaults to 'project', no project given
    assert result.exit_code != 0
    assert "project" in result.output
    assert "Traceback" not in result.output


def test_cli_stats_reports_pending(tmp_path, monkeypatch):
    from mnemo.adapters.store.in_memory_repository import InMemoryMemoryRepository
    from mnemo.domain.memory import Memory

    runner, app = _runner_and_app(tmp_path, monkeypatch)
    runner.invoke(app, ["store", "embedded note", "--project", "api"])  # CLI embeds inline

    # Inject a vector-less (pending) memory into the same store file.
    repo = InMemoryMemoryRepository(path=str(tmp_path / "memory.json"))
    repo.add(Memory.create("not embedded yet", project="api"))  # no vector → pending

    stats = json.loads(runner.invoke(app, ["stats"]).stdout)
    assert stats["total"] == 2
    assert stats["pending"] == 1  # only the vector-less one


def test_cli_recall_groups_a_projects_memory_by_type(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMO_LOG_LEVEL", "ERROR")  # keep timing/model logs off stdout
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    runner.invoke(app, ["store", "use jwt", "--type", "decision", "--project", "api"])
    runner.invoke(app, ["store", "fixed a race", "--type", "debug", "--project", "api"])
    runner.invoke(app, ["store", "other thing", "--type", "decision", "--project", "other"])

    result = runner.invoke(app, ["recall", "api", "auth"])
    assert result.exit_code == 0, result.output
    bundle = json.loads(result.stdout)
    assert bundle["project"] == "api"
    assert bundle["query"] == "auth"
    assert bundle["total"] == 2  # the 'other' project is excluded
    assert bundle["summary"] is None  # no generator configured → structured bundle only
    by_type = {section["type"]: section for section in bundle["sections"]}
    assert set(by_type) == {"decision", "debug"}
    assert len(by_type["decision"]["memories"]) == 1


def test_cli_recall_rejects_a_blank_query(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    result = runner.invoke(app, ["recall", "api", "   "])  # whitespace-only query
    assert result.exit_code != 0
    assert "query" in result.output
    assert "Traceback" not in result.output


def test_cli_recall_reports_missing_model_dep_without_a_traceback(tmp_path, monkeypatch):
    # The reranker/generator adapters raise an actionable RuntimeError when their
    # optional extra is absent; the CLI must show the message, not a stack trace.
    runner, app = _runner_and_app(tmp_path, monkeypatch)

    class _Boom:
        def execute(self, **_kwargs):
            raise RuntimeError(
                'the recall generator needs llama-cpp-python — install the model extra '
                '(pip install "mnemo[recall]") or set MNEMO_GENERATOR=off'
            )

    class _Container:
        recall = _Boom()

    monkeypatch.setattr(
        "mnemo.adapters.cli.app.build_container", lambda *a, **k: _Container()
    )

    result = runner.invoke(app, ["recall", "api", "auth"])
    assert result.exit_code == 1
    assert "llama-cpp-python" in result.output
    assert "Traceback" not in result.output
