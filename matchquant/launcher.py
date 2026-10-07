"""Installed-entry boundary for the MatchQuant CLI with narrow dependency diagnostics.

The console script (``matchquant``) and ``python -m matchquant`` both enter
through :func:`main`. Its only addition over calling
:mod:`matchquant.cli` directly is a narrow, honest diagnosis when numpy or
pandas cannot be imported from the running interpreter: instead of a bare
traceback, it points the user at the correct venv and the documented offline
install steps. Any other import error is re-raised unchanged; nothing else is
caught, and the CLI import architecture is untouched.
"""

from __future__ import annotations

import sys

_CORE_DEPENDENCIES = ("numpy", "pandas")


def _dependency_message(name: str) -> str:
    return (
        f"MatchQuant cannot import its core dependency '{name}' from this Python interpreter.\n"
        "This usually means the wrong interpreter is running (not the project venv), or the\n"
        "offline installation step was never run in this environment.\n"
        "Fix: run MatchQuant with the venv interpreter created from the delivery, and repeat the\n"
        "documented offline install in that exact venv, for example:\n"
        "  <your-venv>\\Scripts\\python.exe -m pip install --no-index --find-links <wheelhouse> "
        "matchquant==0.1.0\n"
        "See the '安装' / 'Quick start' section of the MatchQuant README for the exact steps."
    )


def _import_cli():
    """Import the real CLI; kept as a module function so the diagnosis branch is testable."""

    from .cli import main as cli_main

    return cli_main


def main() -> None:
    try:
        cli_main = _import_cli()
    except ModuleNotFoundError as exc:
        missing = (getattr(exc, "name", "") or "").split(".")[0]
        if missing not in _CORE_DEPENDENCIES:
            raise
        print(_dependency_message(missing), file=sys.stderr)
        raise SystemExit(1) from None
    cli_main()


__all__ = ["main"]
