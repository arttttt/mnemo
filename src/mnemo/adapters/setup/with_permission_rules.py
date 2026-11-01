"""A client installer that also writes mnemo's permission rules for that client.

Wraps any ClientInstaller: wiring the server is the wrapped installer's job, and the
rules come after it. They are written even when the wiring fails: `claude mcp add`
refuses a server that is already there, and a re-run of setup on a machine that has
mnemo wired is how the rules get added. The result is "failed" when either step is.
"""
from __future__ import annotations

from mnemo.adapters.setup.client_installer import ClientInstaller
from mnemo.adapters.setup.install_result import InstallResult
from mnemo.adapters.setup.permission_rules import PermissionRules, RulesNotWritten


class WithPermissionRules:
    def __init__(self, installer: ClientInstaller, rules: PermissionRules) -> None:
        self._installer = installer
        self._rules = rules

    @property
    def name(self) -> str:
        return self._installer.name

    def detect(self) -> bool:
        return self._installer.detect()

    def describe(self) -> str:
        return f"{self._installer.describe()}; {self._rules.describe()}"

    def install(self) -> InstallResult:
        wired = self._installer.install()
        status = wired.status
        try:
            note = self._rules.apply()
        except RulesNotWritten as refused:
            status = "failed"
            note = (
                f"permission rules not written: {refused.reason}. "
                f"Add them to {self._rules.target} by hand:\n{refused.snippet}"
            )
        message = "; ".join(part for part in (wired.message, note) if part)
        return InstallResult(wired.client, status, f"{wired.target}; {self._rules.target}", message)
