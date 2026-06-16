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
