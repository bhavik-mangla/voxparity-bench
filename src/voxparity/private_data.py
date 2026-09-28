"""Loader for the private annotation tables that ship with the held-out bank.

Some analyses need per-item knowledge of held-out items: hand codes, rubric
overrides, the confirmed hard-tail golds, the legacy-item list. Those tables
name held-out items, variants and tools, so they live in JSON files under
:func:`voxparity.paths.private_dir` (``VOXPARITY_PRIVATE_DATA``) and never in
the code. Organisers holding the bank regenerate everything; a public checkout
imports every module cleanly and gets a clear :class:`PrivateDataUnavailable`
when an analysis actually needs a table it does not have.

Two ways to load:

* :func:`load` returns the data, or a :class:`Missing` placeholder that raises
  :class:`PrivateDataUnavailable` on first use (for module-level tables, so the
  module still imports);
* :func:`load` with ``default=`` returns the default instead (for code that has a
  well-defined public behaviour without the table).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

from voxparity.paths import private_dir

_UNSET: Any = object()


class PrivateDataUnavailable(FileNotFoundError):
    """A private (held-out bank) table is needed and not present."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(
            f"private data not available: {name!r} is not under {private_dir()}. It ships "
            "with the held-out bank, not with the code; set VOXPARITY_PRIVATE_DATA (or "
            "VXP_BANK) to run this analysis."
        )


class Missing:
    """Stands in for an absent private table; any use raises PrivateDataUnavailable."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", name)

    def _raise(self, *_: Any, **__: Any) -> NoReturn:
        raise PrivateDataUnavailable(object.__getattribute__(self, "_name"))

    def __getattr__(self, attr: str) -> Any:
        self._raise()

    __getitem__ = __iter__ = __len__ = __contains__ = __bool__ = __call__ = _raise

    def __repr__(self) -> str:
        return f"<missing private data {object.__getattribute__(self, '_name')!r}>"


def path(name: str) -> Path:
    return private_dir() / name


def available(name: str) -> bool:
    return path(name).is_file()


def load(
    name: str,
    convert: Callable[[Any], Any] | None = None,
    *,
    default: Any = _UNSET,
) -> Any:
    """Load ``<private_dir>/<name>`` (JSON). Absent: ``default`` if given, else Missing."""
    p = path(name)
    if not p.is_file():
        return Missing(name) if default is _UNSET else default
    data = json.loads(p.read_text())
    return convert(data) if convert else data


def require(*names: str) -> None:
    """Raise PrivateDataUnavailable unless every named table is present."""
    for n in names:
        if not available(n):
            raise PrivateDataUnavailable(n)
