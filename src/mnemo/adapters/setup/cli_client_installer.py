"""Wire a client that ships its own `mcp add` command (Claude Code, Codex, Kimi).

Shelling out to the client's official command is more robust than hand-editing
its config: the client owns its format/scopes. The runner is injected so this is
tested by asserting the exact argv, without the real client binary.
"""
from __future__ import annotations

from mnemo.adapters.setup.command_runner import CommandRunner
from mnemo.adapters.setup.install_result import InstallResult


class CliClientInstaller:
    def __init__(
        self,
        name: str,
        binary: str,
        add_prefix: list[str],
        server_name: str,
        connector_argv: list[str],
        runner: CommandRunner,
    ) -> None:
        self._name = name
        self._binary = binary
        self._add_prefix = add_prefix
        self._server_name = server_name
        self._connector_argv = connector_argv
        self._runner = runner

    @property
    def name(self) -> str:
        return self._name

    def detect(self) -> bool:
        import shutil

        return shutil.which(self._binary) is not None

    def _argv(self) -> list[str]:
        # e.g. claude mcp add --transport stdio --scope user mnemo -- <connector...>
        return [self._binary, *self._add_prefix, self._server_name, "--", *self._connector_argv]

    def describe(self) -> str:
        return f"run `{' '.join(self._argv())}`"

    def install(self) -> InstallResult:
        argv = self._argv()
        code = self._runner.run(argv)
        status = "ok" if code == 0 else "failed"
        message = "" if code == 0 else f"`{self._binary}` exited with {code}"
        return InstallResult(self._name, status, " ".join(argv), message)
