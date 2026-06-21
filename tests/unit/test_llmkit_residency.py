"""ResidencyManager is a bounded pool of independent runtime instances.

size=1 reduces to the old single-instance behaviour (resident keeps the model, transient
frees it on idle); size>1 leases distinct instances for real parallelism.
"""
from __future__ import annotations

import threading

import pytest

from llmkit.lifecycle.manager import ResidencyManager
from llmkit.lifecycle.residency import Resident, Transient


class _FakeRuntime:
    def __init__(self) -> None:
        self.loads = 0
        self.unloads = 0
        self.loaded = False

    def load(self) -> None:
        self.loads += 1
        self.loaded = True

    def unload(self) -> None:
        self.unloads += 1
        self.loaded = False


def _counting_factory():
    """A factory that mints a fresh _FakeRuntime per call and records them all."""
    made: list[_FakeRuntime] = []

    def factory() -> _FakeRuntime:
        runtime = _FakeRuntime()
        made.append(runtime)
        return runtime

    return factory, made


# --- size=1 reduces to the old single-instance behaviour ---


def test_resident_size_one_loads_once_and_keeps_loaded():
    rt = _FakeRuntime()
    manager = ResidencyManager(lambda: rt, Resident(), size=1)
    with manager.use() as runtime:
        assert runtime.loaded
    assert rt.loaded  # still loaded after the use scope
    with manager.use():
        pass
    assert rt.loads == 1 and rt.unloads == 0  # never reloaded, never freed


def test_transient_size_one_frees_after_use_and_reloads_next_time():
    rt = _FakeRuntime()
    manager = ResidencyManager(lambda: rt, Transient(), size=1)
    with manager.use():
        assert rt.loaded
    assert not rt.loaded
    assert rt.loads == 1 and rt.unloads == 1
    with manager.use():
        assert rt.loaded
    assert rt.loads == 2  # reloaded for the second use


# --- pool behaviour (size > 1) ---


def test_concurrent_leases_get_distinct_instances():
    factory, made = _counting_factory()
    manager = ResidencyManager(factory, Resident(), size=3)
    seen: list[int] = []
    barrier = threading.Barrier(3)

    def worker() -> None:
        with manager.use() as rt:
            barrier.wait(timeout=3.0)  # all three hold a lease at once → real parallelism
            seen.append(id(rt))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert len(set(seen)) == 3  # three distinct instances ran concurrently
    assert len(made) == 3 and all(rt.loaded for rt in made)


def test_pool_blocks_when_exhausted_then_proceeds_on_return():
    manager = ResidencyManager(_counting_factory()[0], Resident(), size=1)
    held, release, proceeded = (threading.Event() for _ in range(3))

    def holder() -> None:
        with manager.use():
            held.set()
            release.wait(timeout=3.0)

    def waiter() -> None:
        with manager.use():
            proceeded.set()

    h = threading.Thread(target=holder)
    h.start()
    assert held.wait(2.0)
    w = threading.Thread(target=waiter)
    w.start()
    assert not proceeded.wait(0.3)  # blocked: the single instance is leased
    release.set()
    assert proceeded.wait(2.0)  # freed → the waiter gets the instance
    h.join(2.0)
    w.join(2.0)


def test_resident_reuses_a_warm_instance_serially():
    factory, made = _counting_factory()
    manager = ResidencyManager(factory, Resident(), size=2)
    with manager.use():
        pass
    with manager.use():
        pass
    assert len(made) == 1  # serial reuse never minted a second instance


def test_transient_unloads_each_returned_instance():
    factory, made = _counting_factory()
    manager = ResidencyManager(factory, Transient(), size=2)
    with manager.use():
        pass
    with manager.use():
        pass
    assert len(made) == 2  # each transient lease minted then unloaded its own instance
    assert all(rt.unloads == 1 and not rt.loaded for rt in made)


def test_load_failure_releases_the_permit_and_does_not_shrink_the_pool():
    calls = {"n": 0}

    def factory() -> _FakeRuntime:
        runtime = _FakeRuntime()
        calls["n"] += 1
        if calls["n"] == 1:
            def boom() -> None:
                raise RuntimeError("load failed")
            runtime.load = boom  # the first instance fails to load
        return runtime

    manager = ResidencyManager(factory, Resident(), size=1)
    with pytest.raises(RuntimeError):
        with manager.use():
            pass
    # the failed load freed the permit; the pool is not shrunk and a fresh lease works
    with manager.use() as rt:
        assert rt.loaded
    assert calls["n"] == 2


def test_close_unloads_idle_instances_and_refuses_new_leases():
    factory, made = _counting_factory()
    manager = ResidencyManager(factory, Resident(), size=2)
    with manager.use():
        pass

    manager.close()

    assert made and all(not rt.loaded for rt in made)  # every warm instance unloaded
    with pytest.raises(RuntimeError):
        with manager.use():
            pass
