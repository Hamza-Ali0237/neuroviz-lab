"""Errors that explain what was expected, what was received, and how to fix it."""

from __future__ import annotations

import difflib
from collections.abc import Iterable


class NvzError(Exception):
    """Base class for nvzlab errors."""


class ShapeError(NvzError, ValueError):
    """An op received arrays whose shapes do not fit together."""


class UnknownNameError(NvzError, KeyError):
    """A model, dataset or op name is not registered."""

    def __str__(self) -> str:  # KeyError would quote the message
        return self.args[0]


def unknown_name(kind: str, name: str, choices: Iterable[str]) -> UnknownNameError:
    """Build an error for an unknown ``kind`` name, suggesting the closest match."""
    choices = sorted(choices)
    close = difflib.get_close_matches(name, choices, n=1, cutoff=0.5)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    listing = ", ".join(repr(c) for c in choices) or "(none)"
    return UnknownNameError(f"Unknown {kind} {name!r}.{hint} Available {kind}s: {listing}.")


def shape_error(op: str, expected: str, received: str, fix: str) -> ShapeError:
    """Build a shape error naming the op, the expectation, the actual shapes and a fix."""
    return ShapeError(f"{op}: expected {expected}, received {received}. Fix: {fix}")


def check_shape(cond: bool, op: str, expected: str, received: str, fix: str) -> None:
    """Raise ``ShapeError`` unless ``cond`` holds."""
    if not cond:
        raise shape_error(op, expected, received, fix)
