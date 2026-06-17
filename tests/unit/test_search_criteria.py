"""SearchCriteria invariants — a project-scoped search must name its project."""
import pytest

from mnemo.application.search_criteria import SearchCriteria


def test_project_scope_without_a_project_is_rejected():
    with pytest.raises(ValueError) as exc:
        SearchCriteria(scope="project")  # project defaults to None
    message = str(exc.value)
    assert "scope='project'" in message  # names the offending scope
    assert "global" in message and "all" in message  # points at the alternatives


def test_project_scope_with_a_project_is_valid():
    criteria = SearchCriteria(scope="project", project="api")
    assert criteria.project == "api"


def test_global_and_all_scopes_need_no_project():
    assert SearchCriteria(scope="global").project is None
    assert SearchCriteria(scope="all").project is None


def test_created_after_accepts_iso_date_and_datetime():
    assert SearchCriteria(scope="all", created_after="2026-06-01").created_after
    assert SearchCriteria(scope="all", created_after="2026-06-01T00:00:00+00:00").created_after
    assert SearchCriteria(scope="all", created_after="2026-06-01T00:00:00Z").created_after


def test_created_after_rejects_non_iso():
    with pytest.raises(ValueError) as exc:
        SearchCriteria(scope="all", created_after="last tuesday")
    assert "ISO-8601" in str(exc.value)


@pytest.mark.parametrize("scope", ["global", "all"])
def test_project_is_rejected_with_global_or_all_scope(scope):
    # scope is authoritative for these — a project would be silently dropped, so the
    # contradiction is rejected rather than ignored.
    with pytest.raises(ValueError) as exc:
        SearchCriteria(scope=scope, project="api")
    message = str(exc.value)
    assert f"scope='{scope}'" in message and "project" in message
