"""PipelineContext — a generic typed slot map: copy-on-write, typed get, filled set."""
from __future__ import annotations

import pytest

from mnemo.application.pipeline.context import PipelineContext
from mnemo.application.pipeline.errors import PipelineError
from mnemo.application.pipeline.slot import Slot

COUNT: Slot[int] = Slot("count")
ITEMS: Slot[tuple[int, ...]] = Slot("items")


def test_set_is_copy_on_write_leaving_the_original_untouched():
    base = PipelineContext.empty()
    extended = base.set(COUNT, 1)
    assert not base.has(COUNT)  # the original is unchanged
    assert extended.get(COUNT) == 1


def test_get_on_an_unproduced_slot_raises():
    with pytest.raises(PipelineError) as exc:
        PipelineContext.empty().get(COUNT)
    assert "count" in str(exc.value)


def test_filled_reports_the_produced_slots():
    ctx = PipelineContext.empty().set(COUNT, 1)
    assert ctx.filled == frozenset({COUNT.name})


def test_an_empty_value_still_counts_as_produced():
    # produced-but-empty must differ from absent, or a stage that legitimately
    # produces nothing would look like it never ran.
    ctx = PipelineContext.empty().set(ITEMS, ())
    assert ITEMS.name in ctx.filled
    assert ctx.get(ITEMS) == ()
