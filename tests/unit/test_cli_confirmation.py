"""Unit tests for the reusable destructive-confirmation component."""
import pytest

typer = pytest.importorskip("typer")

from mnemo.adapters.cli.confirmation import confirm_or_abort


def test_assume_yes_confirms_without_prompting(monkeypatch):
    # --yes must bypass the prompt entirely (non-interactive use).
    monkeypatch.setattr(typer, "prompt", lambda *a, **k: pytest.fail("must not prompt when assume_yes"))
    confirm_or_abort("wipe everything?", assume_yes=True)  # returns, no prompt, no raise


def test_yes_answer_proceeds(monkeypatch):
    monkeypatch.setattr(typer, "prompt", lambda *a, **k: "y")
    confirm_or_abort("wipe everything?")  # no raise == the command proceeds


def test_no_answer_aborts(monkeypatch):
    monkeypatch.setattr(typer, "prompt", lambda *a, **k: "n")
    with pytest.raises(typer.Abort):
        confirm_or_abort("wipe everything?")


def test_reasks_until_a_valid_yes_or_no(monkeypatch):
    # The component blocks until the answer is explicit: invalid answers re-prompt.
    answers = iter(["maybe", "", "yes"])  # two invalid answers, then a clear yes
    monkeypatch.setattr(typer, "prompt", lambda *a, **k: next(answers))
    confirm_or_abort("wipe everything?")  # loops past the invalid answers, then proceeds
