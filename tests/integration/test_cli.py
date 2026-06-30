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


# --- JSON contract (--json) ----------------------------------------------------------
# These exercise the stable machine-readable payloads agents/scripts parse. The default
# (no --json) human-readable view is covered in its own section further down.

def test_cli_version_reports_installed_distribution(monkeypatch):
    import mnemo.adapters.cli.app as cli_app

    def fail_container(*_args, **_kwargs):
        raise AssertionError("version must not build the application container")

    monkeypatch.setattr(cli_app, "build_container", fail_container)
    runner = testing.CliRunner()
    expected = package_version("mnemo")

    result = runner.invoke(cli_app.app, ["version"])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == expected  # default: the bare version number


def test_cli_version_json_wraps_the_number(monkeypatch):
    import mnemo.adapters.cli.app as cli_app

    result = testing.CliRunner().invoke(cli_app.app, ["version", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"version": package_version("mnemo")}


def test_cli_store_then_search(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)

    stored = runner.invoke(
        app, ["store", "JWT refresh rotation", "--type", "decision", "--project", "api", "--json"]
    )
    assert stored.exit_code == 0, stored.output
    memory_id = json.loads(stored.stdout)["id"]

    found = runner.invoke(app, ["search", "jwt rotation", "--project", "api", "--json"])
    assert found.exit_code == 0, found.output
    assert memory_id in found.stdout  # full id surfaces in the JSON payload


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

    result = runner.invoke(app, ["search", "anything", "--scope", "global", "--json"])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "[]"  # empty store, but the command runs end-to-end


def test_cli_browse_lists_memories_without_a_query(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    a = json.loads(runner.invoke(app, ["store", "alpha", "--project", "api", "--json"]).stdout)["id"]
    b = json.loads(runner.invoke(app, ["store", "beta", "--project", "api", "--json"]).stdout)["id"]

    result = runner.invoke(app, ["browse", "--project", "api", "--json"])
    assert result.exit_code == 0, result.output
    hits = json.loads(result.stdout)
    created = [hit["created_at"] for hit in hits]
    assert created == sorted(created, reverse=True)  # newest first
    assert {hit["id"] for hit in hits} == {a, b}
    assert all("score" not in hit for hit in hits)  # browse carries no score
    assert all("topic_key" in hit and hit["status"] == "active" for hit in hits)  # audit fields on hits


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
         "--tag", "auth", "--file", "src/auth.py", "--json"],
    )
    assert stored.exit_code == 0, stored.output
    memory_id = json.loads(stored.stdout)["id"]

    # The metadata was actually set: search filters on tag/file find it...
    by_tag = runner.invoke(
        app, ["search", "jwt rotation", "--project", "api", "--tag", "auth", "--json"]
    )
    assert memory_id in by_tag.stdout
    by_file = runner.invoke(
        app, ["search", "jwt rotation", "--project", "api", "--file", "src/auth.py", "--json"]
    )
    assert memory_id in by_file.stdout

    # ...and a non-matching tag filter excludes it (proves it wasn't silently dropped).
    miss = runner.invoke(
        app, ["search", "jwt rotation", "--project", "api", "--tag", "cache", "--json"]
    )
    assert memory_id not in miss.stdout


def test_cli_delete_purge_and_stats(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)

    one = json.loads(runner.invoke(app, ["store", "one", "--project", "api", "--json"]).stdout)["id"]
    runner.invoke(app, ["store", "two", "--project", "api"])
    runner.invoke(app, ["store", "three", "--project", "other"])

    stats = json.loads(runner.invoke(app, ["stats", "--json"]).stdout)
    assert stats["total"] == 3
    assert stats["pending"] == 0  # the CLI embeds inline (sync scheduler), so nothing pending
    assert json.loads(runner.invoke(app, ["delete", one, "--json"]).stdout)["deleted"] == 1
    assert json.loads(runner.invoke(app, ["purge", "--yes", "--json"]).stdout)["deleted"] == 2
    assert json.loads(runner.invoke(app, ["stats", "--json"]).stdout)["total"] == 0


def test_cli_purge_confirmation_gate(tmp_path, monkeypatch):
    # purge drops EVERYTHING, so it asks first: answering 'n' aborts and keeps the data,
    # answering 'y' goes through. (--yes, tested above, skips the prompt non-interactively.)
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    runner.invoke(app, ["store", "keep me", "--project", "api"])

    aborted = runner.invoke(app, ["purge"], input="n\n")
    assert aborted.exit_code != 0  # declined → aborted, nothing deleted
    assert json.loads(runner.invoke(app, ["stats", "--json"]).stdout)["total"] == 1

    confirmed = runner.invoke(app, ["purge"], input="y\n")
    assert confirmed.exit_code == 0, confirmed.output
    # (the prompt text shares stdout, so verify the effect via stats rather than parsing it)
    assert json.loads(runner.invoke(app, ["stats", "--json"]).stdout)["total"] == 0  # everything gone


def test_cli_delete_cascade_removes_the_whole_lineage(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    # Build a supersede chain v1 <- v2 <- v3 under one topic_key (the CLI embeds inline).
    runner.invoke(app, ["store", "auth v1", "--project", "api", "--topic-key", "auth/model"])
    runner.invoke(app, ["store", "auth v2", "--project", "api", "--topic-key", "auth/model"])
    head = json.loads(
        runner.invoke(
            app, ["store", "auth v3", "--project", "api", "--topic-key", "auth/model", "--json"]
        ).stdout
    )["id"]

    deleted = runner.invoke(app, ["delete", head, "--cascade", "--json"])
    assert deleted.exit_code == 0, deleted.output
    assert json.loads(deleted.stdout)["deleted"] == 3  # head + the two superseded versions

    # the whole topic is gone (without --cascade only the head would have been removed)
    found = runner.invoke(app, ["search", "auth", "--project", "api", "--json"])
    assert json.loads(found.stdout) == []


def test_cli_stats_reports_pending(tmp_path, monkeypatch):
    from mnemo.adapters.embedding.hash_embedder import HashEmbedder
    from mnemo.adapters.store.sqlite_vec_repository import SqliteRepositoryImpl
    from mnemo.domain.memory import Memory

    runner, app = _runner_and_app(tmp_path, monkeypatch)
    runner.invoke(app, ["store", "embedded note", "--project", "api"])  # CLI embeds inline

    # Inject a vector-less (pending) memory into the same store file.
    repo = SqliteRepositoryImpl.open(path=str(tmp_path / "memory.db"), dim=HashEmbedder().dim)
    repo.add(Memory.create("not embedded yet", project="api"))  # no vector → pending

    stats = json.loads(runner.invoke(app, ["stats", "--json"]).stdout)
    assert stats["total"] == 2
    assert stats["pending"] == 1  # only the vector-less one


def test_cli_recall_returns_the_query_relevant_memories_as_light_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMO_LOG_LEVEL", "ERROR")  # keep timing/model logs off stdout
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    runner.invoke(app, ["store", "use jwt", "--type", "decision", "--project", "api"])
    runner.invoke(app, ["store", "fixed a race", "--type", "learning", "--project", "api"])
    runner.invoke(app, ["store", "other thing", "--type", "decision", "--project", "other"])

    result = runner.invoke(app, ["recall", "api", "auth", "--json"])
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


def test_cli_search_fails_fast_on_a_dimension_mismatch(tmp_path, monkeypatch):
    # End-to-end: bake the store at an odd dimension FIRST (the CLI's sqlite_path is
    # MNEMO_DATA_DIR/memory.db), then run the hash embedder (dim 256) against it. The guard
    # must turn the mismatch into a clean BadParameter, not a deep CHECK / sqlite-vec failure.
    pytest.importorskip("sqlite_vec")
    from tests.support.sqlite_store import open_store

    open_store(tmp_path, dim=7)  # tmp_path/memory.db, memories CHECK baked at dim 7
    monkeypatch.setenv("MNEMO_EMBEDDER", "hash")  # dim 256 ≠ 7
    monkeypatch.setenv("MNEMO_RERANKER", "off")
    monkeypatch.setenv("MNEMO_GENERATOR", "off")
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))
    from mnemo.adapters.cli.app import app

    result = testing.CliRunner().invoke(app, ["search", "anything", "--scope", "all"])

    assert result.exit_code != 0
    assert "dimension mismatch" in result.output
    assert "7" in result.output and "256" in result.output  # both dimensions surfaced
    assert "Traceback" not in result.output                 # clean message, no stack trace


def test_cli_get_by_topic_key(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    runner.invoke(app, ["store", "auth v1", "--project", "api", "--topic-key", "auth/model"])
    runner.invoke(app, ["store", "auth v2", "--project", "api", "--topic-key", "auth/model"])

    result = runner.invoke(app, ["get", "--topic-key", "auth/model", "--project", "api", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["content"] == "auth v2" and payload["status"] == "active"
    assert payload["chain_total"] == 2
    assert [e["status"] for e in payload["chain"]] == ["active", "superseded"]


def test_cli_create_project_then_store(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    created = runner.invoke(
        app, ["create-project", "newproj", "--description", "a new one", "--json"]
    )
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

    deleted = runner.invoke(app, ["delete-project", "api", "--json"])
    assert deleted.exit_code == 0, deleted.output
    assert json.loads(deleted.stdout)["slug"] == "api"

    # the project's memory cascaded away with it
    found = runner.invoke(app, ["search", "doomed", "--scope", "all", "--json"])
    assert json.loads(found.stdout) == []


def test_cli_update_and_list_projects(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)  # pre-registers api, other

    updated = runner.invoke(app, ["update-project", "api", "the API service", "--json"])
    assert updated.exit_code == 0, updated.output
    assert json.loads(updated.stdout)["description"] == "the API service"

    listed = runner.invoke(app, ["list-projects", "--json"])
    assert listed.exit_code == 0, listed.output
    slugs = {p["slug"] for p in json.loads(listed.stdout)}
    assert {"api", "other"} <= slugs


def test_cli_create_project_rejects_an_over_budget_description(tmp_path, monkeypatch):
    # The 128-token description cap is enforced through the real CLI -> container -> use case:
    # over-budget exits non-zero with a clean message (not a traceback) and nothing is registered.
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    over = " ".join(["word"] * 129)  # hash embedder counts one token per word -> over the 128 cap
    result = runner.invoke(app, ["create-project", "fresh", "--description", over])
    assert result.exit_code != 0
    assert "128-token limit" in result.output
    assert "Traceback" not in result.output
    # the slug is free to register once the description fits
    assert runner.invoke(app, ["create-project", "fresh", "-d", "a small service"]).exit_code == 0


def test_cli_update_project_rejects_an_over_budget_description(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)  # pre-registers "api"
    result = runner.invoke(app, ["update-project", "api", " ".join(["w"] * 200)])
    assert result.exit_code != 0
    assert "128-token limit" in result.output
    assert "Traceback" not in result.output
    assert runner.invoke(app, ["update-project", "api", "the api service"]).exit_code == 0


# --- Human-readable default (no --json) ----------------------------------------------
# The default view is for a person at a terminal: scannable text, not JSON. These assert
# the default is NOT JSON and carries the right cues.

def test_cli_store_default_is_human_readable(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    result = runner.invoke(app, ["store", "a note", "--type", "decision", "--project", "api"])
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("created — ")  # status verb + id, not a JSON object
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)


def test_cli_search_default_lists_type_and_snippet(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    runner.invoke(
        app,
        ["store", "JWT refresh rotation in httpOnly cookies", "--type", "decision",
         "--project", "api", "--topic-key", "auth/jwt"],
    )

    result = runner.invoke(app, ["search", "jwt rotation", "--project", "api"])
    assert result.exit_code == 0, result.output
    assert "1 hit" in result.stdout
    assert "[decision]" in result.stdout         # type cue
    assert "auth/jwt" in result.stdout           # the durable handle leads the line
    assert "JWT refresh rotation" in result.stdout  # the content snippet


def test_cli_search_default_empty_is_a_clear_note(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    result = runner.invoke(app, ["search", "anything", "--scope", "global"])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "No memories found."


def test_cli_stats_default_is_human_readable(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    runner.invoke(app, ["store", "x", "--type", "rule", "--project", "api"])

    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0, result.output
    assert "memories: 1" in result.stdout
    assert "by type:" in result.stdout
    assert "rule" in result.stdout


def test_cli_get_default_shows_content_and_chain(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    runner.invoke(app, ["store", "auth v1", "--project", "api", "--topic-key", "auth/model"])
    runner.invoke(app, ["store", "auth v2", "--project", "api", "--topic-key", "auth/model"])

    result = runner.invoke(app, ["get", "--topic-key", "auth/model", "--project", "api"])
    assert result.exit_code == 0, result.output
    assert "auth v2" in result.stdout        # untruncated content of the active head
    assert "id: " in result.stdout           # full id is the actionable handle here
    assert "chain (2 of 2):" in result.stdout


def test_cli_recall_default_shows_header_and_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMO_LOG_LEVEL", "ERROR")
    runner, app = _runner_and_app(tmp_path, monkeypatch)
    runner.invoke(app, ["store", "use jwt", "--type", "decision", "--project", "api"])

    result = runner.invoke(app, ["recall", "api", "auth"])
    assert result.exit_code == 0, result.output
    assert "recall: api" in result.stdout
    assert 'query "auth"' in result.stdout
    assert "no generated summary" in result.stdout  # generator is off in tests
    assert "sources:" in result.stdout


def test_cli_list_projects_default_is_human_readable(tmp_path, monkeypatch):
    runner, app = _runner_and_app(tmp_path, monkeypatch)  # pre-registers api, other
    result = runner.invoke(app, ["list-projects"])
    assert result.exit_code == 0, result.output
    assert "projects:" in result.stdout
    assert "api" in result.stdout and "other" in result.stdout
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)
