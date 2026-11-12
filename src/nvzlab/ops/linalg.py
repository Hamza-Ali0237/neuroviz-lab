"""Linear algebra ops: matmul, add, mul, linear."""

from __future__ import annotations

import numpy as np

from nvzlab.core.errors import check_shape
from nvzlab.core.op import op
from nvzlab.ops._util import broadcastable, unbroadcast


@op(
    category="linalg",
    formula=r"C = A B",
    text="Each output entry is the dot product of a row of A with a column of B.",
    shape_rule="(n, k) x (k, m) -> (n, m); (n, k) x (k,) -> (n,)",
    backward_formula=(
        r"\frac{\partial L}{\partial A} = g B^\top,\quad \frac{\partial L}{\partial B} = A^\top g"
    ),
    backward_text=(
        "The upstream gradient g is multiplied by the transpose of the other operand, so "
        "each input's gradient has the input's own shape."
    ),
)
def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix product ``a @ b``.

    Examples
    --------
    >>> from nvzlab import ops
    >>> ops.matmul(np.array([[1.0, 2.0]]), np.array([[3.0], [4.0]]))
    array([[11.]])
    """
    check_shape(
        a.ndim == 2 and b.ndim in (1, 2),
        "matmul",
        "a to be 2-D and b to be 1-D or 2-D",
        f"a {a.shape} and b {b.shape}",
        "reshape the inputs, e.g. x.reshape(1, -1) for a single sample",
    )
    check_shape(
        a.shape[1] == b.shape[0],
        "matmul",
        f"a.shape[1] == b.shape[0] (inner dimensions equal), a is {a.shape}",
        f"b {b.shape}",
        f"give b a first dimension of {a.shape[1]} (or transpose it)",
    )
    return a @ b


@matmul.backward
def _(grad, a, b, out):
    if b.ndim == 1:
        return np.outer(grad, b), a.T @ grad
    return grad @ b.T, a.T @ grad


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
    shape_rule="(n, f) x (f, m) + (m,) -> (n, m); (n, f) x (f,) + (1,) -> (n,)",
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
        x.ndim == 2,
        "linear",
        "x to be 2-D (n_samples, n_features)",
        f"x {x.shape}",
        "reshape a single sample as x.reshape(1, -1)",
    )
    check_shape(
        w.ndim in (1, 2) and w.shape[0] == x.shape[1],
        "linear",
        f"w with shape ({x.shape[1]},) or ({x.shape[1]}, m) to match x {x.shape}",
        f"w {w.shape}",
        f"give w a first dimension of {x.shape[1]}",
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
    return gx, x.T @ grad, unbroadcast(grad, b.shape)
