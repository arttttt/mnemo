"""ResidencyManager — resident holds the model, transient frees it once unused."""
from __future__ import annotations

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

    def work(self) -> str:
        return "ok"


def test_resident_loads_once_and_keeps_loaded():
    rt = _FakeRuntime()
    manager = ResidencyManager(rt, Resident())
    with manager.use() as runtime:
        assert runtime.loaded and runtime.work() == "ok"
    assert rt.loaded  # still loaded after the use scope
    with manager.use():
        pass
    assert rt.loads == 1 and rt.unloads == 0  # never reloaded, never freed


def test_transient_frees_after_use_and_reloads_next_time():
    rt = _FakeRuntime()
    manager = ResidencyManager(rt, Transient())
    with manager.use():
        assert rt.loaded
    assert not rt.loaded
    assert rt.loads == 1 and rt.unloads == 1
    with manager.use():
        assert rt.loaded
    assert rt.loads == 2  # reloaded for the second use


def test_transient_refcount_keeps_loaded_until_the_last_user_leaves():
    rt = _FakeRuntime()
    manager = ResidencyManager(rt, Transient())
    with manager.use():
        with manager.use():
            assert rt.loaded
        assert rt.loaded and rt.unloads == 0  # inner exit must not free it
    assert not rt.loaded and rt.unloads == 1  # outer exit frees it
