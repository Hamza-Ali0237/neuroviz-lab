"""One registry shared by built-in and user-defined ops."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nvzlab.core.errors import NvzError, unknown_name

if TYPE_CHECKING:
    from nvzlab.core.op import Op

_OPS: dict[str, Op] = {}


def register(op: Op) -> Op:
    """Register ``op`` under its name. Built-in names cannot be overwritten."""
    existing = _OPS.get(op.name)
    if existing is not None and existing.origin == "builtin" and op is not existing:
        raise NvzError(
            f"{op.name!r} is a built-in op and cannot be replaced. "
            f"Fix: pick another name, e.g. @nvz.op(name='my_{op.name}')."
        )
    _OPS[op.name] = op
    return op


def get(name: str) -> Op:
    """Return the op registered as ``name``."""
    if name not in _OPS:
        raise unknown_name("op", name, _OPS)
    return _OPS[name]


def names() -> list[str]:
    """Sorted names of every registered op."""
    return sorted(_OPS)
