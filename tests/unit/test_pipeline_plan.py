"""The plan vocabulary — operations are matchable and an empty plan reports empty."""
from __future__ import annotations

from mnemo.application.pipeline.operations import Flag, Insert, Supersede
from mnemo.application.pipeline.plan import Plan
from mnemo.application.pipeline.proposed_memory import ProposedMemory
from mnemo.domain.memory_type import MemoryType
from mnemo.domain.scope import Scope


def _merged() -> ProposedMemory:
    return ProposedMemory(
        content="merged record",
        type=MemoryType.DECISION,
        project="api",
        scope=Scope.PROJECT,
    )


def test_an_empty_plan_reports_empty():
    assert Plan().is_empty


def test_a_plan_carries_its_operations_in_order_for_matching():
    plan = Plan(
        (
            Supersede(source_ids=("a", "b"), replacement=_merged()),
            Insert(record=_merged()),
            Flag(left_id="x", right_id="y"),
        )
    )
    assert not plan.is_empty
    assert [type(op).__name__ for op in plan.operations] == ["Supersede", "Insert", "Flag"]
    assert plan.operations[2].kind == "contradiction"  # default flag kind
