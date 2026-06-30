"""Unit tests for the CLI presentation layer (src/mnemo/adapters/cli/output.py).

The renderers are pure (data in, str out), so they are exercised here directly with
light stand-ins (SimpleNamespace) for the result dataclasses — no CLI runner, no store.
The end-to-end JSON-vs-human wiring is covered in tests/integration/test_cli.py.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from mnemo.adapters.cli import output


def _hit(**overrides):
    base = dict(
        id="ea8abf22089543053ce9ec37e5acfcf618713238",
        type="decision",
        content="MCP agent surface — behavior via params, not a tool per type/op.",
        created_at="2026-06-25T20:58:47.906469+00:00",
        topic_key="api/surface",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --- render() / json_option() --------------------------------------------------------

def test_render_json_emits_parseable_indented_payload(capsys):
    output.render({"a": 1, "b": ["x"]}, "human view", as_json=True)
    out = capsys.readouterr().out
    assert json.loads(out) == {"a": 1, "b": ["x"]}
    assert "\n" in out  # indented, not compact


def test_render_human_emits_the_text_not_json(capsys):
    output.render({"a": 1}, "human view", as_json=False)
    assert capsys.readouterr().out.strip() == "human view"


def test_render_json_keeps_non_ascii_unescaped(capsys):
    output.render({"q": "auth поток"}, "", as_json=True)
    assert "поток" in capsys.readouterr().out  # ensure_ascii=False


def test_json_option_is_a_fresh_off_by_default_flag():
    first, second = output.json_option(), output.json_option()
    assert first is not second  # a factory, not one shared mutable instance
    assert first.default is False
    assert "--json" in first.param_decls


# --- format_hits (search / browse) ---------------------------------------------------

def test_format_hits_empty_is_a_clear_note():
    assert output.format_hits([]) == "No memories found."


def test_format_hits_counts_and_pluralizes():
    assert output.format_hits([_hit()]).startswith("1 hit\n")
    assert output.format_hits([_hit(), _hit()]).startswith("2 hits\n")


def test_format_hits_leads_with_topic_key_and_trims_the_date():
    rendered = output.format_hits([_hit()])
    assert "[decision] api/surface" in rendered
    assert "2026-06-25" in rendered
    assert "20:58:47" not in rendered  # only the date, not the time


def test_format_hits_falls_back_to_a_short_id_when_no_topic_key():
    rendered = output.format_hits([_hit(topic_key=None)])
    assert "id:ea8abf220895" in rendered            # 12-char prefix, the only handle
    assert "ea8abf22089543053" not in rendered      # not the full 40-char id


def test_format_hits_collapses_whitespace_and_elides_long_content():
    long = "word " * 60  # ~300 chars after collapse, well over the snippet width
    rendered = output.format_hits([_hit(content="alpha\n\n   beta\t" + long)])
    snippet = rendered.splitlines()[-1].strip()
    assert snippet.startswith("alpha beta word")  # newlines/tabs collapsed to single spaces
    assert snippet.endswith("…")                  # elided
    assert len(snippet) <= output._SNIPPET_WIDTH


# --- format_remember -----------------------------------------------------------------

def test_format_remember_shows_status_and_id():
    result = SimpleNamespace(id="abc123", status="created")
    assert output.format_remember(result) == "created — abc123"


# --- format_get ----------------------------------------------------------------------

def _get_result(**overrides):
    base = dict(
        id="head01",
        type="decision",
        scope="project",
        project="mnemo",
        content="the full, untruncated body that get is expected to show in one piece",
        related_files=["src/a.py", "src/b.py"],
        created_at="2026-06-25T10:00:00+00:00",
        topic_key="api/surface",
        status="active",
        supersedes="prev01",
        chain=[
            SimpleNamespace(id="head01", status="active", created_at="2026-06-25T10:00:00+00:00"),
            SimpleNamespace(id="prev01", status="superseded", created_at="2026-06-20T10:00:00+00:00"),
        ],
        chain_total=2,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_format_get_shows_full_content_metadata_and_chain():
    rendered = output.format_get(_get_result())
    assert "the full, untruncated body that get is expected to show in one piece" in rendered
    assert "id: head01" in rendered
    assert "project: mnemo" in rendered
    assert "files: src/a.py, src/b.py" in rendered
    assert "supersedes: prev01" in rendered
    assert "chain (2 of 2):" in rendered
    assert "active" in rendered and "superseded" in rendered
    assert "prev01" in rendered  # full chain ids so older versions are gettable


def test_format_get_omits_absent_optional_fields_and_marks_no_topic_key():
    rendered = output.format_get(
        _get_result(topic_key=None, supersedes=None, related_files=[], project=None)
    )
    assert "(no topic_key)" in rendered
    assert "supersedes:" not in rendered
    assert "files:" not in rendered
    assert "project:" not in rendered


def test_format_get_reports_a_partial_chain_window():
    rendered = output.format_get(_get_result(chain_total=7))
    assert "chain (2 of 7):" in rendered  # showing a capped window of a longer lineage


# --- format_recall -------------------------------------------------------------------

def test_format_recall_with_a_summary_lists_sources():
    payload = {
        "project": "mnemo",
        "query": "auth",
        "total": 2,
        "summary": "Auth uses JWT with refresh rotation.",
        "sources": [{"id": "m1", "type": "decision"}, {"id": "m2", "type": "learning"}],
    }
    rendered = output.format_recall(payload)
    assert 'query "auth"' in rendered
    assert "2 memories" in rendered
    assert "Auth uses JWT with refresh rotation." in rendered
    assert "  • [decision] m1" in rendered
    assert "  • [learning] m2" in rendered


def test_format_recall_without_a_summary_notes_the_generator_is_off():
    payload = {"project": "mnemo", "query": "auth", "total": 1, "summary": None,
               "sources": [{"id": "m1", "type": "decision"}]}
    rendered = output.format_recall(payload)
    assert "no generated summary" in rendered
    assert "1 memory" in rendered  # singular


# --- format_stats --------------------------------------------------------------------

def test_format_stats_shows_totals_and_a_sorted_breakdown():
    rendered = output.format_stats(
        {"total": 5, "pending": 1, "by_type": {"rule": 2, "decision": 3}}
    )
    assert "memories: 5  (pending: 1)" in rendered
    lines = rendered.splitlines()
    # by_type is sorted by name: decision before rule
    assert lines.index("  decision  3") < lines.index("  rule      2")


def test_format_stats_with_no_memories_omits_the_breakdown():
    rendered = output.format_stats({"total": 0, "pending": 0, "by_type": {}})
    assert rendered == "memories: 0  (pending: 0)"


# --- format_reindex ------------------------------------------------------------------

def test_format_reindex_dry_run():
    rendered = output.format_reindex({"memories": 12, "target_dim": 1024, "dry_run": True})
    assert "would re-embed 12 memories at dim 1024" in rendered
    assert "dry run" in rendered


def test_format_reindex_completed_reports_restart():
    assert "service restarted: yes" in output.format_reindex(
        {"reindexed": 12, "dim": 1024, "service_restarted": True}
    )
    assert "service restarted: no" in output.format_reindex(
        {"reindexed": 0, "dim": 1024, "service_restarted": False}
    )


# --- format_deletion -----------------------------------------------------------------

def test_format_deletion_pluralizes_and_handles_purge():
    assert output.format_deletion(SimpleNamespace(deleted=1)) == "1 memory deleted"
    assert output.format_deletion(SimpleNamespace(deleted=3)) == "3 memories deleted"
    assert output.format_deletion(SimpleNamespace(deleted=2), purged=True) == (
        "2 memories deleted; project registry reset"
    )


# --- projects ------------------------------------------------------------------------

def test_format_project_created_with_and_without_description():
    assert output.format_project_created(
        SimpleNamespace(slug="web", description="the web app")
    ) == "created project 'web' — the web app"
    assert output.format_project_created(
        SimpleNamespace(slug="web", description=None)
    ) == "created project 'web'"


def test_format_project_deleted_and_updated():
    assert output.format_project_deleted(SimpleNamespace(slug="web")) == (
        "deleted project 'web' and all its memories"
    )
    assert output.format_project_updated(
        SimpleNamespace(slug="web", description="new desc")
    ) == "updated project 'web': new desc"


def test_format_projects_empty_and_listed():
    assert output.format_projects([]) == "No projects registered."
    rendered = output.format_projects([
        SimpleNamespace(slug="web", description="the web app"),
        SimpleNamespace(slug="api", description=None),
    ])
    assert rendered.startswith("2 projects:")
    assert "web  the web app" in rendered
    assert "api  (no description)" in rendered  # missing description filled in
