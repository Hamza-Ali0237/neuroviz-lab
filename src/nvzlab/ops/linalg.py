"""Linear algebra ops: matmul, add, mul, linear, transpose, scale."""

from __future__ import annotations

import numpy as np

from nvzlab.core.errors import check_shape
from nvzlab.core.op import op
from nvzlab.ops._util import broadcastable, unbroadcast


@op(
    category="linalg",
    formula=r"C = A B",
    text="Each output entry is the dot product of a row of A with a column of B.",
    shape_rule=(
        "(n, k) x (k, m) -> (n, m); (n, k) x (k,) -> (n,); "
        "batched (b, n, k) x (b, k, m) -> (b, n, m)"
    ),
    backward_formula=(
        r"\frac{\partial L}{\partial A} = g B^\top,\quad \frac{\partial L}{\partial B} = A^\top g"
    ),
    backward_text=(
        "The upstream gradient g is multiplied by the transpose of the other operand, so "
        "each input's gradient has the input's own shape."
    ),
)
def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix product ``a @ b``; leading (batch) dimensions broadcast like ``np.matmul``.

    Examples
    --------
    >>> from nvzlab import ops
    >>> ops.matmul(np.array([[1.0, 2.0]]), np.array([[3.0], [4.0]]))
    array([[11.]])
    """
    check_shape(
        a.ndim >= 2 and (b.ndim >= 2 or (b.ndim == 1 and a.ndim == 2)),
        "matmul",
        "a to be at least 2-D and b to be at least 2-D (or 1-D with a 2-D a)",
        f"a {a.shape} and b {b.shape}",
        "reshape the inputs, e.g. x.reshape(1, -1) for a single sample",
    )
    check_shape(
        a.shape[-1] == b.shape[0 if b.ndim == 1 else -2],
        "matmul",
        f"the last dimension of a to equal the second-to-last of b, a is {a.shape}",
        f"b {b.shape}",
        f"give b a (second-to-last) dimension of {a.shape[-1]} (or transpose it)",
    )
    check_shape(
        b.ndim == 1 or broadcastable(a.shape[:-2], b.shape[:-2]),
        "matmul",
        "batch dimensions that broadcast together",
        f"a {a.shape} and b {b.shape}",
        "match the leading dimensions (or use 1)",
    )
    return a @ b


@matmul.backward
def _(grad, a, b, out):
    if b.ndim == 1:
        return np.outer(grad, b), a.T @ grad
    ga = grad @ np.swapaxes(b, -1, -2)
    gb = np.swapaxes(a, -1, -2) @ grad
    return unbroadcast(ga, a.shape), unbroadcast(gb, b.shape)


@op(
    category="linalg",
    formula=r"c = a + b",
    text="Element-wise sum; a smaller operand is repeated (broadcast) over the larger one.",
    shape_rule="shapes broadcast together, e.g. (n, m) + (m,) -> (n, m)",
    backward_formula=(
        r"\frac{\partial L}{\partial a} = g,\quad \frac{\partial L}{\partial b} = g"
        r"\ \text{(summed over broadcast axes)}"
    ),
    backward_text=(
        "Addition passes the upstream gradient g straight through to both operands. An "
        "operand that was broadcast receives the sum of g over the repeated axes."
    ),
)
def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise addition with NumPy broadcasting.

    Examples
    --------
    >>> from nvzlab import ops
    >>> ops.add(np.array([[1.0, 2.0]]), np.array([10.0, 20.0]))
    array([[11., 22.]])
    """
    check_shape(
        broadcastable(a.shape, b.shape),
        "add",
        "shapes that broadcast together",
        f"a {a.shape} and b {b.shape}",
        "make the trailing dimensions match (or be 1)",
    )
    return a + b


@add.backward
def _(grad, a, b, out):
    return unbroadcast(grad, a.shape), unbroadcast(grad, b.shape)


@op(
    category="linalg",
    formula=r"c = a \odot b",
    text="Element-wise product; a smaller operand is repeated (broadcast) over the larger one.",
    shape_rule="shapes broadcast together, e.g. (n, m) * (m,) -> (n, m)",
    backward_formula=(
        r"\frac{\partial L}{\partial a} = g \odot b,\quad \frac{\partial L}{\partial b} = g \odot a"
        r"\ \text{(summed over broadcast axes)}"
    ),
    backward_text=(
        "Each operand's gradient is the upstream gradient g times the other operand. An "
        "operand that was broadcast receives the sum over the repeated axes."
    ),
)
def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise multiplication with NumPy broadcasting.

    Examples
    --------
    >>> from nvzlab import ops
    >>> ops.mul(np.array([[1.0, 2.0]]), np.array([10.0, 20.0]))
    array([[10., 40.]])
    """
    check_shape(
        broadcastable(a.shape, b.shape),
        "mul",
        "shapes that broadcast together",
        f"a {a.shape} and b {b.shape}",
        "make the trailing dimensions match (or be 1)",
    )
    return a * b


@mul.backward
def _(grad, a, b, out):
    return unbroadcast(grad * b, a.shape), unbroadcast(grad * a, b.shape)


@op(
    category="linalg",
    formula=r"z = X w + b",
    text=(
        "A weighted sum of the inputs plus a bias: every feature is scaled by its weight "
        "and the results are added."
    ),
    shape_rule="(..., f) x (f, m) + (m,) -> (..., m); (n, f) x (f,) + (1,) -> (n,)",
    backward_formula=(
        r"\frac{\partial L}{\partial X} = g W^\top,\quad \frac{\partial L}{\partial W} = X^\top g,"
        r"\quad \frac{\partial L}{\partial b} = \sum_i g_i"
    ),
    backward_text=(
        "The upstream gradient g flows to the inputs through the weights and to the "
        "weights through the inputs; the bias gets g summed over the samples."
    ),
)
def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Affine map ``x @ w + b``.

    Examples
    --------
    >>> from nvzlab import ops
    >>> ops.linear(np.array([[1.0, 2.0]]), np.array([0.5, -0.5]), np.array([1.0]))
    array([0.5])
    """
    check_shape(
        x.ndim >= 2,
        "linear",
        "x to be at least 2-D (n_samples, ..., n_features)",
        f"x {x.shape}",
        "reshape a single sample as x.reshape(1, -1)",
    )
    check_shape(
        (w.ndim == 2 or (w.ndim == 1 and x.ndim == 2)) and w.shape[0] == x.shape[-1],
        "linear",
        f"w with shape ({x.shape[-1]},) or ({x.shape[-1]}, m) to match x {x.shape}",
        f"w {w.shape}",
        f"give w a first dimension of {x.shape[-1]}",
    )
    z = x @ w
    check_shape(
        broadcastable(z.shape, b.shape) and np.broadcast_shapes(z.shape, b.shape) == z.shape,
        "linear",
        f"b to broadcast over x @ w with shape {z.shape}",
        f"b {b.shape}",
        f"use b with shape ({z.shape[-1]},)" if z.ndim == 2 else "use b with shape (1,)",
    )
    return z + b


@linear.backward
def _(grad, x, w, b, out):
    gx = np.outer(grad, w) if w.ndim == 1 else grad @ w.T
    lead = list(range(x.ndim - 1))
    return gx, np.tensordot(x, grad, axes=(lead, lead)), unbroadcast(grad, b.shape)


@op(
    category="shape",
    formula=r"y_{\pi(i)} = x_i",
    text="Reorders the axes; no value changes, only where it sits.",
    shape_rule="shape permuted by `axes`, e.g. (n, T, d) with axes (0, 2, 1) -> (n, d, T)",
    backward_formula=r"\frac{\partial L}{\partial x} = \mathrm{transpose}(g, \pi^{-1})",
    backward_text="The upstream gradient is permuted back with the inverse axis order.",
)
def transpose(x: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """Permute the axes of ``x`` (``np.transpose(x, axes)``).

    Examples
    --------
    >>> from nvzlab import ops
    >>> ops.transpose(np.zeros((2, 3, 4)), axes=(0, 2, 1)).shape
    (2, 4, 3)
    """
    axes = tuple(int(a) for a in axes)
    check_shape(
        sorted(axes) == list(range(x.ndim)),
        "transpose",
        f"axes to be a permutation of 0..{x.ndim - 1} for x {x.shape}",
        f"axes={axes}",
        "list every axis exactly once, e.g. axes=(0, 2, 1) to swap the last two",
    )
    return np.transpose(x, axes)


@transpose.backward
def _(grad, x, out, axes):
    return np.transpose(grad, np.argsort(axes))


@op(
    category="linalg",
    formula=r"y = c\, x",
    text="Multiplies every element by the same constant c.",
    shape_rule="same shape as the input",
    backward_formula=r"\frac{\partial L}{\partial x} = c\, g",
    backward_text="The upstream gradient is multiplied by the same constant.",
)
def scale(x: np.ndarray, c: float) -> np.ndarray:
    """Multiply by a constant (recorded as the attr ``c``).

    Examples
    --------
    >>> from nvzlab import ops
    >>> ops.scale(np.array([2.0, 4.0]), c=0.5)
    array([1., 2.])
    """
    check_shape(
        isinstance(c, int | float) and not isinstance(c, bool),
        "scale",
        "c to be a plain number",
        f"c={c!r}",
        "pass a float, e.g. scale(x, c=1 / 8 ** 0.5)",
    )
    return x * c


@scale.backward
def _(grad, x, out, c):
    return grad * c
