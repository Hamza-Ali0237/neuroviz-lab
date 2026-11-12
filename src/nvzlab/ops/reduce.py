"""Reduction ops: mean, sum."""

from __future__ import annotations

import numpy as np

from nvzlab.core.errors import check_shape
from nvzlab.core.op import op


def _check_axis(name: str, x: np.ndarray, axis: int | None) -> None:
    check_shape(
        axis is None or -x.ndim <= axis < x.ndim,
        name,
        f"axis=None or an axis in [{-x.ndim}, {x.ndim - 1}] for shape {x.shape}",
        f"axis={axis}",
        "pass an axis that exists",
    )


def _spread(grad: np.ndarray, x: np.ndarray, axis: int | None) -> np.ndarray:
    if axis is None:
        return np.broadcast_to(grad, x.shape).copy()
    return np.broadcast_to(np.expand_dims(grad, axis), x.shape).copy()


@op(
    category="reduce",
    formula=r"\bar{x} = \frac{1}{n}\sum_i x_i",
    text="Average of the values (over everything, or along one axis).",
    shape_rule="(...) -> scalar, or the input shape with `axis` removed",
    backward_formula=r"\frac{\partial L}{\partial x_i} = \frac{g}{n}",
    backward_text=(
        "Every averaged element receives an equal share g / n of the upstream gradient."
    ),
)
def mean(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Arithmetic mean.

    Examples
    --------
    >>> from nvzlab import ops
    >>> float(ops.mean(np.array([1.0, 3.0])))
    2.0
    """
    _check_axis("mean", x, axis)
    return np.asarray(np.mean(x, axis=axis))


@mean.backward
def _(grad, x, out, axis=None):
    n = x.size if axis is None else x.shape[axis]
    return _spread(grad, x, axis) / n


@op(
    category="reduce",
    formula=r"s = \sum_i x_i",
    text="Sum of the values (over everything, or along one axis).",
    shape_rule="(...) -> scalar, or the input shape with `axis` removed",
    backward_formula=r"\frac{\partial L}{\partial x_i} = g",
    backward_text="Every summed element receives a copy of the upstream gradient g.",
)
def sum(x: np.ndarray, axis: int | None = None) -> np.ndarray:  # noqa: A001
    """Sum of elements.

    Examples
    --------
    >>> from nvzlab import ops
    >>> float(ops.sum(np.array([1.0, 3.0])))
    4.0
    """
    _check_axis("sum", x, axis)
    return np.asarray(np.sum(x, axis=axis))


@sum.backward
def _(grad, x, out, axis=None):
    return _spread(grad, x, axis)
