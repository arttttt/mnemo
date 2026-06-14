"""parse_selection turns a user's list pick into chosen 0-based indices."""
from mnemo.adapters.setup.selection import parse_selection


def test_all_and_star_select_everything():
    assert parse_selection("all", 3) == [0, 1, 2]
    assert parse_selection("*", 2) == [0, 1]


def test_none_and_empty_select_nothing():
    assert parse_selection("none", 3) == []
    assert parse_selection("", 3) == []


def test_comma_list_maps_to_zero_based():
    assert parse_selection("1,3", 3) == [0, 2]


def test_spaces_dupes_and_out_of_range_are_handled():
    assert parse_selection("3 3 9 2", 3) == [1, 2]


def test_non_numeric_tokens_are_ignored():
    assert parse_selection("x, 2 ,y", 3) == [1]
