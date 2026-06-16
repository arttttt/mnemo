"""stop_service — terminate the shared on-demand service by its pidfile."""
import os
import shlex
import subprocess
import sys
import time

from mnemo.adapters.mcp.run_paths import run_dir
from mnemo.adapters.mcp.service_control import stop_service
from mnemo.infrastructure.config import Config


def _config(tmp_path, monkeypatch) -> Config:
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMO_EMBEDDER", "hash")
    return Config.from_env()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _spawn_detached_dummy(pid_file) -> int:
    # A process the test does NOT parent: a backgrounding shell reparents it to init, so
    # after we kill it the OS reaps it and it reads as gone — exactly like the real detached
    # service. It records its own pid in the pidfile and ignores SIGHUP so the shell exiting
    # can't take it down (only stop_service's SIGTERM/SIGKILL should).
    code = (
        "import os, signal, time;"
        "signal.signal(signal.SIGHUP, signal.SIG_IGN);"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()));"
        "time.sleep(30)"
    )
    subprocess.run(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(code)} &", shell=True, check=True
    )
    for _ in range(200):  # up to ~4s for the child to record its pid
        text = pid_file.read_text().strip() if pid_file.exists() else ""
        if text:
            return int(text)
        time.sleep(0.02)
    raise RuntimeError("dummy service did not start")


def test_stops_a_running_service_and_clears_the_pidfile(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch)
    run_dir(config).mkdir(parents=True, exist_ok=True)
    pid_file = run_dir(config) / "service.pid"
    pid = _spawn_detached_dummy(pid_file)
    assert _alive(pid)

    assert stop_service(config) is True

    deadline = time.time() + 3
    while _alive(pid) and time.time() < deadline:
        time.sleep(0.02)
    assert not _alive(pid)
    assert not pid_file.exists()


def test_no_pidfile_is_a_noop(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch)
    assert stop_service(config) is False


def test_stale_pidfile_is_cleared(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch)
    run_dir(config).mkdir(parents=True, exist_ok=True)
    pid_file = run_dir(config) / "service.pid"
    pid_file.write_text("999999")  # a pid that is not running

    assert stop_service(config) is False
    assert not pid_file.exists()
