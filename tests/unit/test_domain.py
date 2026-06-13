from mnemo.domain.constants import GLOBAL_PROJECT
from mnemo.domain.hashing import content_hash, normalize
from mnemo.domain.memory import Memory
from mnemo.domain.memory_type import MemoryType
from mnemo.domain.scope import Scope


def test_normalize_collapses_whitespace_and_lowercases():
    assert normalize("  Hello   World\n") == "hello world"


def test_content_hash_is_stable_under_normalization():
    assert content_hash("Hello   World") == content_hash("hello world")


def test_create_defaults_to_working_notes_and_active():
    memory = Memory.create("a note")
    assert memory.type is MemoryType.WORKING_NOTES
    assert memory.scope is Scope.PROJECT
    assert memory.status == "active"
    assert memory.hash == content_hash("a note")


def test_global_scope_forces_global_project():
    memory = Memory.create("a rule", type="rule", scope="global", project="ignored")
    assert memory.scope is Scope.GLOBAL
    assert memory.project == GLOBAL_PROJECT


def test_register_duplicate_increments_count():
    memory = Memory.create("x")
    memory.register_duplicate()
    assert memory.duplicate_count == 1


def test_mark_superseded():
    memory = Memory.create("x")
    memory.mark_superseded()
    assert memory.status == "superseded"
