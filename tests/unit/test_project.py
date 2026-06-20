import pytest

from mnemo.domain.project import Project


def test_create_stamps_created_at_and_defaults_description():
    project = Project.create("pa-kmp")
    assert project.slug == "pa-kmp"
    assert project.description is None
    assert project.created_at  # an ISO timestamp was stamped


def test_create_keeps_description():
    assert Project.create("pa-kmp", "Personal assistant, KMP").description == (
        "Personal assistant, KMP"
    )


def test_create_rejects_empty_slug():
    with pytest.raises(ValueError):
        Project.create("")
    with pytest.raises(ValueError):
        Project.create("   ")
