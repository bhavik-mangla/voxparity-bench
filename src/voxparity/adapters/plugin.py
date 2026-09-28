"""Bring-your-own-agent (BYOA) drivers: load a user class as a SessionDriver.

``voxparity run --driver python:<module_or_path>:<ClassName>`` imports the class,
instantiates it with any ``--driver-arg key=value`` pairs as string keyword
arguments, and checks it against the SessionDriver contract (adapters/base.py)
before a single cell runs. The class may subclass ``SessionDriver`` or merely
implement the same surface (``name``, ``capabilities``, ``respond``;
``preflight`` optional).

``<module_or_path>`` is either an importable dotted module (``mycorp.voxdriver``)
or a path to a ``.py`` file (``examples/byoa_echo_driver.py``). The last ``:``
separates the class name, so paths containing ``:`` are not supported.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import re
import sys
from pathlib import Path
from typing import Any

from voxparity.adapters.base import DriverCapabilities, SessionContext, SessionDriver
from voxparity.schemas.result import TurnResult

PREFIX = "python:"


class PluginDriverError(ValueError):
    """A BYOA driver could not be loaded or does not satisfy the contract."""


def parse_driver_args(pairs: list[str] | None) -> dict[str, str]:
    """``["url=http://x", "timeout=5"]`` -> ``{"url": "http://x", "timeout": "5"}``."""
    out: dict[str, str] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        key = key.strip()
        if not sep or not key.isidentifier():
            raise PluginDriverError(f"--driver-arg must be key=value with a valid name: {pair!r}")
        out[key] = value
    return out


def _import_target(target: str) -> Any:
    path = Path(target)
    if target.endswith(".py") or "/" in target or path.is_file():
        if not path.is_file():
            raise PluginDriverError(f"driver file not found: {target}")
        resolved = path.resolve()
        digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:10]
        stem = re.sub(r"\W", "_", resolved.stem)
        mod_name = f"voxparity_byoa_{stem}_{digest}"
        if mod_name in sys.modules:
            return sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(mod_name, resolved)
        if spec is None or spec.loader is None:
            raise PluginDriverError(f"cannot import driver file: {target}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            sys.modules.pop(mod_name, None)
            raise PluginDriverError(f"error importing {target}: {e}") from e
        return module
    try:
        return importlib.import_module(target)
    except ImportError as e:
        raise PluginDriverError(f"cannot import driver module {target!r}: {e}") from e


def load_driver_class(spec: str) -> type:
    """Resolve ``[python:]<module_or_path>:<ClassName>`` to a class object."""
    body = spec[len(PREFIX) :] if spec.startswith(PREFIX) else spec
    target, sep, cls_name = body.rpartition(":")
    if not sep or not target or not cls_name:
        raise PluginDriverError(f"driver spec {spec!r} must be python:<module_or_path>:<ClassName>")
    module = _import_target(target)
    cls = getattr(module, cls_name, None)
    if cls is None:
        raise PluginDriverError(f"{target} has no attribute {cls_name!r}")
    if not inspect.isclass(cls):
        raise PluginDriverError(f"{target}:{cls_name} is not a class")
    missing = [attr for attr in ("respond", "capabilities") if not hasattr(cls, attr)]
    if missing:
        raise PluginDriverError(
            f"{cls_name} does not implement the SessionDriver contract: "
            f"missing {', '.join(missing)}"
        )
    if not callable(getattr(cls, "respond", None)):
        raise PluginDriverError(f"{cls_name}.respond must be a method respond(ctx) -> TurnResult")
    if inspect.isabstract(cls):
        abstract = ", ".join(sorted(getattr(cls, "__abstractmethods__", ())))
        raise PluginDriverError(f"{cls_name} is abstract (unimplemented: {abstract})")
    return cls


class _ProtocolAdapter(SessionDriver):
    """Wraps a duck-typed driver that does not subclass SessionDriver."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.name = inner.name
        self.experiment = getattr(inner, "experiment", None)

    @property
    def capabilities(self) -> DriverCapabilities:
        return self._inner.capabilities  # type: ignore[no-any-return]

    def preflight(self) -> None:
        fn = getattr(self._inner, "preflight", None)
        if callable(fn):
            fn()

    def respond(self, ctx: SessionContext) -> TurnResult:
        return self._inner.respond(ctx)  # type: ignore[no-any-return]


def load_plugin_driver(spec: str, driver_args: dict[str, str] | None = None) -> SessionDriver:
    """Import, instantiate and validate a BYOA driver. Raises PluginDriverError
    with a message meant for the command line."""
    cls = load_driver_class(spec)
    kwargs = dict(driver_args or {})
    try:
        inst = cls(**kwargs)
    except (TypeError, ValueError) as e:
        raise PluginDriverError(
            f"cannot construct {cls.__name__}({', '.join(kwargs) or ''}): {e}"
        ) from e
    name = getattr(inst, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise PluginDriverError(
            f"{cls.__name__}.name must be a non-empty string (it is recorded on every row)"
        )
    caps = inst.capabilities
    if not isinstance(caps, DriverCapabilities):
        raise PluginDriverError(
            f"{cls.__name__}.capabilities must return voxparity.adapters.base.DriverCapabilities, "
            f"got {type(caps).__name__}"
        )
    if isinstance(inst, SessionDriver):
        return inst
    return _ProtocolAdapter(inst)


def run_id_slug(driver: SessionDriver) -> str:
    """A filesystem-safe run-id fragment for a plugin driver (spec may hold a path)."""
    return "python_" + re.sub(r"[^A-Za-z0-9._+-]", "_", driver.name)
