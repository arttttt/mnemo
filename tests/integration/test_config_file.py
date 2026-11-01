"""Client config files are replaced whole, keep their mode, and stay symlinks."""
import os
import stat

import pytest

from mnemo.adapters.setup import config_file
from mnemo.adapters.setup.config_file import replace_file


def test_a_new_file_is_created_with_its_directory(tmp_path):
    path = tmp_path / ".claude" / "settings.json"

    replace_file(path, b"{}\n")

    assert path.read_bytes() == b"{}\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert os.listdir(path.parent) == ["settings.json"]  # no temporary file left behind


def test_an_existing_file_keeps_its_mode(tmp_path):
    path = tmp_path / "settings.json"
    path.write_bytes(b"old")
    path.chmod(0o640)

    replace_file(path, b"new")

    assert path.read_bytes() == b"new"
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_a_symlinked_config_stays_a_symlink(tmp_path):
    dotfiles = tmp_path / "dotfiles" / "settings.json"
    dotfiles.parent.mkdir()
    dotfiles.write_bytes(b"old")
    link = tmp_path / "settings.json"
    link.symlink_to(dotfiles)

    replace_file(link, b"new")

    assert link.is_symlink()
    assert dotfiles.read_bytes() == b"new"


def test_a_failed_write_leaves_the_old_file_and_no_temporary_one(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_bytes(b"old")

    def broken_replace(source, destination):
        raise OSError("disk full")

    monkeypatch.setattr(config_file.os, "replace", broken_replace)
    with pytest.raises(OSError):
        replace_file(path, b"new")

    assert path.read_bytes() == b"old"
    assert os.listdir(tmp_path) == ["settings.json"]
