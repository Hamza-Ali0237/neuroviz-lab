"""Linear algebra ops: matmul, add, linear."""

from __future__ import annotations

import numpy as np

from nvzlab.core.errors import check_shape
from nvzlab.core.op import op
from nvzlab.ops._util import broadcastable, unbroadcast


@op(
    category="builtin",
    formula=r"C = A B",
    text="Each output entry is the dot product of a row of A with a column of B.",
    shape_rule="(n, k) x (k, m) -> (n, m); (n, k) x (k,) -> (n,)",
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
    category="builtin",
    formula=r"c = a + b",
    text="Element-wise sum; a smaller operand is repeated (broadcast) over the larger one.",
    shape_rule="shapes broadcast together, e.g. (n, m) + (m,) -> (n, m)",
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
    category="builtin",
    formula=r"z = X w + b",
    text=(
        "A weighted sum of the inputs plus a bias: every feature is scaled by its weight "
        "and the results are added."
    ),
    shape_rule="(n, f) x (f, m) + (m,) -> (n, m); (n, f) x (f,) + (1,) -> (n,)",
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
