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


@pytest.mark.parametrize(
    "bad",
    [
        "My Project",   # spaces
        "API",          # uppercase
        "foo_bar",      # underscore
        "foo bar",      # space
        "-leading",     # leading hyphen
        "trailing-",    # trailing hyphen
        "double--hyphen",
        "weird!",       # punctuation
        "../escape",    # path-ish
        "__global__",   # the reserved sentinel is not a user-creatable slug
    ],
)
def test_create_rejects_non_kebab_slug(bad):
    with pytest.raises(ValueError):
        Project.create(bad)


@pytest.mark.parametrize("good", ["api", "checkout-api", "svc-a", "p", "x1", "a-b-c"])
def test_create_accepts_kebab_slug(good):
    assert Project.create(good).slug == good
