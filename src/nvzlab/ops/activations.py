"""Activation ops: sigmoid, relu, softmax, tanh."""

from __future__ import annotations

import numpy as np

from nvzlab.core.errors import check_shape
from nvzlab.core.op import op


@op(
    category="activation",
    formula=r"\sigma(z) = \frac{1}{1 + e^{-z}}",
    text=(
        "Squashes any number into (0, 1), so it can be read as a probability. Large "
        "positive inputs go to 1, large negative ones to 0."
    ),
    shape_rule="same shape as the input",
    backward_formula=r"\frac{\partial L}{\partial z} = g \cdot \sigma(z)(1-\sigma(z))",
    backward_text=(
        "The upstream gradient g is scaled by the sigmoid's slope, which is largest at "
        "z = 0 (0.25) and vanishes for very large or very small z."
    ),
)
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic sigmoid, computed in a numerically stable way.

    Examples
    --------
    >>> from nvzlab import ops
    >>> ops.sigmoid(np.array([0.0]))
    array([0.5])
    """
    return 0.5 * (1.0 + np.tanh(0.5 * z))


@sigmoid.backward
def _(grad, z, out):
    return grad * out * (1.0 - out)


@op(
    category="activation",
    formula=r"\mathrm{relu}(x) = \max(0, x)",
    text="Keeps positive values and sets negative ones to zero.",
    shape_rule="same shape as the input",
    backward_formula=r"\frac{\partial L}{\partial x} = g \cdot \mathbb{1}[x > 0]",
    backward_text=(
        "The upstream gradient g passes through unchanged where the input was positive "
        "and is blocked (set to 0) where the input was negative."
    ),
)
def relu(x: np.ndarray) -> np.ndarray:
    """Rectified linear unit.

    Examples
    --------
    >>> from nvzlab import ops
    >>> ops.relu(np.array([-1.0, 2.0]))
    array([0., 2.])
    """
    return np.maximum(x, 0.0)


@relu.backward
def _(grad, x, out):
    return grad * (x > 0)


@op(
    category="activation",
    formula=r"\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}",
    text=(
        "Turns a vector of scores into probabilities that sum to 1 along the chosen axis. "
        "The maximum is subtracted first for numerical stability (the result is "
        "unchanged)."
    ),
    shape_rule="same shape as the input; sums to 1 along `axis`",
    backward_formula=(
        r"\frac{\partial L}{\partial x_i} = s_i \left( g_i - \sum_j g_j s_j \right),"
        r"\quad s = \mathrm{softmax}(x)"
    ),
    backward_text=(
        "Each output depends on every input, so the upstream gradient g is first "
        "centred by its probability-weighted mean and then scaled by the output s."
    ),
)
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax along ``axis``.

    Examples
    --------
    >>> from nvzlab import ops
    >>> ops.softmax(np.array([0.0, 0.0]))
    array([0.5, 0.5])
    """
    check_shape(
        -x.ndim <= axis < x.ndim,
        "softmax",
        f"axis in [{-x.ndim}, {x.ndim - 1}] for an input of shape {x.shape}",
        f"axis={axis}",
        "pass an axis that exists, e.g. axis=-1 for the last one",
    )
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


@softmax.backward
def _(grad, x, out, axis=-1):
    return out * (grad - (grad * out).sum(axis=axis, keepdims=True))


@op(
    category="activation",
    formula=r"\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}",
    text="Squashes any number into (-1, 1), keeping its sign; zero stays zero.",
    shape_rule="same shape as the input",
    backward_formula=r"\frac{\partial L}{\partial x} = g \cdot (1 - \tanh(x)^2)",
    backward_text=(
        "The upstream gradient g is scaled by the slope 1 - tanh(x)^2, which is 1 at x = 0 "
        "and vanishes for large |x|."
    ),
)
def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent.

    Examples
    --------
    >>> from nvzlab import ops
    >>> ops.tanh(np.array([0.0]))
    array([0.])
    """
    return np.tanh(x)


@tanh.backward
def _(grad, x, out):
    return grad * (1.0 - out * out)
