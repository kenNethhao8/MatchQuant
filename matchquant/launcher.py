"""Installed-entry boundary for the MatchQuant CLI with narrow dependency diagnostics.

The console script (``matchquant``) and ``python -m matchquant`` both enter
through :func:`main`. Because the package ``__init__`` is lazy (PEP 562), the
first heavy numpy/pandas import happens inside :func:`_import_cli` — after the
narrow boundary checks here:

1. a ``find_spec`` precheck locates numpy/pandas without importing them, so a
   missing core dependency is diagnosed before any real import runs;
2. if the location check passed but the actual import still fails, the
   exception chain is inspected for a :class:`ModuleNotFoundError` naming a
   core dependency (pandas wraps a missing numpy as a plain ``ImportError``);
3. anything else is re-raised unchanged: unknown import errors keep their real
   type and chain, and no other programming error is misdiagnosed.
"""

from __future__ import annotations

import importlib.util
import sys

_CORE_DEPENDENCIES = ("numpy", "pandas")


def _dependency_message(missing: list[str]) -> str:
    return (
        f"MatchQuant cannot import its core dependenc{'y' if len(missing) == 1 else 'ies'} "
        f"from this Python interpreter: {', '.join(missing)}.\n"
        "This usually means the wrong interpreter is running (not the project venv), or the\n"
        "offline installation step was never run in this environment.\n"
        "Fix: run MatchQuant with the venv interpreter created from the delivery, and repeat the\n"
        "documented offline install in that exact venv, for example:\n"
        "  <your-venv>\\Scripts\\python.exe -m pip install --no-index --find-links <wheelhouse> "
        "matchquant==0.1.0\n"
        "See the '安装' / 'Quick start' section of the MatchQuant README for the exact steps."
    )


def _missing_core_dependencies() -> list[str]:
    missing: list[str] = []
    for name in _CORE_DEPENDENCIES:
        try:
            located = importlib.util.find_spec(name)
        except ImportError:
            missing.append(name)
            continue
        if located is None:
            missing.append(name)
    return missing


def _core_dependency_in_chain(exc: BaseException) -> str | None:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModuleNotFoundError):
            root = (getattr(current, "name", "") or "").split(".")[0]
            if root in _CORE_DEPENDENCIES:
                return root
        current = current.__cause__ or current.__context__
    return None


def _import_cli():
    """Import the real CLI; kept as a module function so the diagnosis branch is testable."""

    from .cli import main as cli_main

    return cli_main


def main() -> None:
    missing = _missing_core_dependencies()
    if missing:
        print(_dependency_message(missing), file=sys.stderr)
        raise SystemExit(1)
    try:
        cli_main = _import_cli()
    except ImportError as exc:
        culprit = _core_dependency_in_chain(exc)
        if culprit is None:
            raise
        print(_dependency_message([culprit]), file=sys.stderr)
        raise SystemExit(1) from None
    cli_main()


__all__ = ["main"]
