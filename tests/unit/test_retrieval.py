"""Retrieval validates that its two ranking inputs come as a pair: a search supplies
text+vector, a browse neither; exactly one without the other is a caller bug, rejected
early with a clear error instead of crashing deep in the store's dense leg."""
import pytest

from mnemo.application.retrieval import Retrieval
from mnemo.application.search_criteria import SearchCriteria

_CRITERIA = SearchCriteria(scope="all")


def test_search_request_carries_both_text_and_vector():
    request = Retrieval(criteria=_CRITERIA, text="redis cache", vector=[0.1, 0.2, 0.3])
    assert request.text == "redis cache" and request.vector == [0.1, 0.2, 0.3]


def test_browse_request_carries_neither():
    request = Retrieval(criteria=_CRITERIA)  # filter-only browse
    assert request.text is None and request.vector is None


def test_text_without_a_vector_is_rejected():
    with pytest.raises(ValueError, match="text without vector"):  # names the missing side
        Retrieval(criteria=_CRITERIA, text="redis cache", vector=None)


def test_vector_without_text_is_rejected():
    with pytest.raises(ValueError, match="vector without text"):  # names the missing side
        Retrieval(criteria=_CRITERIA, text=None, vector=[0.1, 0.2, 0.3])
