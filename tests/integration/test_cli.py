import json
from importlib.metadata import version as package_version

import pytest

testing = pytest.importorskip("typer.testing")


def _runner_and_app(tmp_path, monkeypatch):
    pytest.importorskip("sqlite_vec")
    monkeypatch.setenv("MNEMO_EMBEDDER", "hash")
    monkeypatch.setenv("MNEMO_RERANKER", "off")    # keep tests offline: no model download
    monkeypatch.setenv("MNEMO_GENERATOR", "off")
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))
    from mnemo.adapters.cli.app import app
    from mnemo.infrastructure.composition import build_container

    # The gate requires registered projects; pre-register the ones these tests use.
    # Persisted in the SQLite store, so each CLI invocation's fresh container sees them.
    container = build_container()
    for slug in ("api", "other"):
        container.create_project.execute(slug)
    return testing.CliRunner(), app


def test_cli_version_reports_installed_distribution(monkeypatch):
    import mnemo.adapters.cli.app as cli_app

    def fail_container(*_args, **_kwargs):
        raise AssertionError("version must not build the application container")

    monkeypatch.setattr(cli_app, "build_container", fail_container)
    runner = testing.CliRunner()
    expected = package_version("mnemo")

    result = runner.invoke(cli_app.app, ["version"])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == expected


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


def test_cli_store_project_scope_without_project_fails_cleanly(tmp_path, monkeypatch):
    # The write path enforces the scope↔project contract too: a project-scoped store
    # with no project exits non-zero with a message, not a silently unreachable row.
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    result = runner.invoke(app, ["store", "orphan note"])  # --scope defaults to 'project'
    assert result.exit_code != 0
    assert "project" in result.output
    assert "Traceback" not in result.output


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


def test_cli_delete_purge_and_stats(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)

    one = json.loads(runner.invoke(app, ["store", "one", "--project", "api"]).stdout)["id"]
    runner.invoke(app, ["store", "two", "--project", "api"])
    runner.invoke(app, ["store", "three", "--project", "other"])

    stats = json.loads(runner.invoke(app, ["stats"]).stdout)
    assert stats["total"] == 3
    assert stats["pending"] == 0  # the CLI embeds inline (sync scheduler), so nothing pending
    assert json.loads(runner.invoke(app, ["delete", one]).stdout)["deleted"] == 1
    assert json.loads(runner.invoke(app, ["purge"]).stdout)["deleted"] == 2
    assert json.loads(runner.invoke(app, ["stats"]).stdout)["total"] == 0


def test_cli_stats_reports_pending(tmp_path, monkeypatch):
    from mnemo.adapters.embedding.hash_embedder import HashEmbedder
    from mnemo.adapters.store.sqlite_vec_repository import SqliteRepositoryImpl
    from mnemo.domain.memory import Memory

    runner, app = _runner_and_app(tmp_path, monkeypatch)
    runner.invoke(app, ["store", "embedded note", "--project", "api"])  # CLI embeds inline

    # Inject a vector-less (pending) memory into the same store file.
    repo = SqliteRepositoryImpl.open(path=str(tmp_path / "memory.db"), dim=HashEmbedder().dim)
    repo.add(Memory.create("not embedded yet", project="api"))  # no vector → pending

    stats = json.loads(runner.invoke(app, ["stats"]).stdout)
    assert stats["total"] == 2
    assert stats["pending"] == 1  # only the vector-less one


def test_cli_recall_returns_the_query_relevant_memories_as_light_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMO_LOG_LEVEL", "ERROR")  # keep timing/model logs off stdout
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    runner.invoke(app, ["store", "use jwt", "--type", "decision", "--project", "api"])
    runner.invoke(app, ["store", "fixed a race", "--type", "learning", "--project", "api"])
    runner.invoke(app, ["store", "other thing", "--type", "decision", "--project", "other"])

    result = runner.invoke(app, ["recall", "api", "auth"])
    assert result.exit_code == 0, result.output
    bundle = json.loads(result.stdout)
    assert bundle["project"] == "api"
    assert bundle["query"] == "auth"
    assert bundle["total"] == 2  # the 'other' project is excluded
    assert bundle["summary"] is None  # no generator configured → structured bundle only
    # sources are light references — id + type, never the memory content
    assert sorted(source["type"] for source in bundle["sources"]) == ["decision", "learning"]
    assert all("content" not in source for source in bundle["sources"])
    assert "sections" not in bundle


def test_cli_recall_rejects_a_blank_query(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    result = runner.invoke(app, ["recall", "api", "   "])  # whitespace-only query
    assert result.exit_code != 0
    assert "query" in result.output
    assert "Traceback" not in result.output


def test_cli_recall_reports_a_broken_required_runtime_without_a_traceback(tmp_path, monkeypatch):
    # A broken/incomplete install is still possible even though the generator runtime
    # is required metadata; the CLI must surface the actionable message cleanly.
    runner, app = _runner_and_app(tmp_path, monkeypatch)

    class _Boom:
        def execute(self, **_kwargs):
            raise RuntimeError(
                "llama-cpp-python is a required mnemo dependency but is not importable — "
                "reinstall mnemo"
            )

    # The guarded recall path reads these before recall runs; a None dimension means a
    # fresh store, so the dimension guard is a no-op and the broken generator is what surfaces.
    class _Queue:
        def current_dim(self):
            return None

    class _Embedder:
        dim = 1024

    class _Container:
        recall = _Boom()
        embedding_queue = _Queue()
        embedder = _Embedder()

    monkeypatch.setattr(
        "mnemo.adapters.cli.app.build_container", lambda *a, **k: _Container()
    )

    result = runner.invoke(app, ["recall", "api", "auth"])
    assert result.exit_code == 1
    assert "llama-cpp-python" in result.output
    assert "Traceback" not in result.output


def test_cli_create_project_then_store(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    created = runner.invoke(app, ["create-project", "newproj", "--description", "a new one"])
    assert created.exit_code == 0, created.output
    assert json.loads(created.stdout)["slug"] == "newproj"

    # A write to the freshly-registered project now passes the gate.
    stored = runner.invoke(app, ["store", "note in newproj", "--project", "newproj"])
    assert stored.exit_code == 0, stored.output


def test_cli_create_project_rejects_a_bad_slug(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    result = runner.invoke(app, ["create-project", "Bad Slug"])  # spaces + uppercase
    assert result.exit_code != 0
    assert "kebab-case" in result.output
    assert "Traceback" not in result.output


def test_cli_delete_project_cascades(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    runner.invoke(app, ["store", "doomed note", "--project", "api"])

    deleted = runner.invoke(app, ["delete-project", "api"])
    assert deleted.exit_code == 0, deleted.output
    assert json.loads(deleted.stdout)["slug"] == "api"

    # the project's memory cascaded away with it
    found = runner.invoke(app, ["search", "doomed", "--scope", "all"])
    assert "doomed note" not in found.stdout


def test_cli_update_and_list_projects(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)  # pre-registers api, other

    updated = runner.invoke(app, ["update-project", "api", "the API service"])
    assert updated.exit_code == 0, updated.output
    assert json.loads(updated.stdout)["description"] == "the API service"

    listed = runner.invoke(app, ["list-projects"])
    assert listed.exit_code == 0, listed.output
    slugs = {p["slug"] for p in json.loads(listed.stdout)}
    assert {"api", "other"} <= slugs
