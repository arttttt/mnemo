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


def test_created_after_normalizes_a_non_utc_offset_to_utc():
    # 12:00+03:00 is 09:00 UTC; stored created_at is UTC, so the bound must be normalized
    # or the string comparison the SQL store uses would mis-order it.
    criteria = SearchCriteria(scope="all", created_after="2026-06-19T12:00:00+03:00")
    assert criteria.created_after == "2026-06-19T09:00:00+00:00"


def test_created_after_naive_input_is_taken_as_utc():
    assert (
        SearchCriteria(scope="all", created_after="2026-06-19T12:00:00").created_after
        == "2026-06-19T12:00:00+00:00"
    )


def test_created_after_filters_by_instant_not_by_string():
    from mnemo.domain.memory import Memory

    memory = Memory.create("note", project="api")
    memory.created_at = "2026-06-19T10:00:00+00:00"  # 10:00 UTC
    # bound 12:00+03:00 == 09:00 UTC → the 10:00 memory is after it → kept
    assert SearchCriteria(scope="all", created_after="2026-06-19T12:00:00+03:00").matches(memory)
    # bound 12:00+00:00 == 12:00 UTC → the 10:00 memory is before it → dropped
    assert not SearchCriteria(scope="all", created_after="2026-06-19T12:00:00+00:00").matches(memory)


@pytest.mark.parametrize("scope", ["global", "all"])
def test_project_is_rejected_with_global_or_all_scope(scope):
    # scope is authoritative for these — a project would be silently dropped, so the
    # contradiction is rejected rather than ignored.
    with pytest.raises(ValueError) as exc:
        SearchCriteria(scope=scope, project="api")
    message = str(exc.value)
    assert f"scope='{scope}'" in message and "project" in message
