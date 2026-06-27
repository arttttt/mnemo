"""scaled_pool: the rerank over-fetch grows WITH the requested k (floored + capped), so a
small page does not read the whole store and a large one still has rerank headroom — not a
flat constant."""
from mnemo.application.search.retrieve_stage import scaled_pool


def test_scaled_pool_grades_with_k_within_floor_and_cap():
    assert scaled_pool(1) == 20   # floored at the minimum (not 5)
    assert scaled_pool(5) == 25   # 5 * factor
    assert scaled_pool(10) == 50  # 10 * factor, exactly the cap
    assert scaled_pool(20) == 50  # capped, not 100
    assert scaled_pool(4) == 20   # the floor still wins at the low end
