"""Op base class and the ``@nvz.op`` decorator.

An op is a plain function on NumPy arrays. Outside a recording it just computes; inside a
``Tape`` every call is recorded as a node so it can be replayed and differentiated.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np

from nvzlab.core import registry
from nvzlab.core.tape import current_tape

Explanation = dict[str, Any]


class Op:
    """Base class for ops (advanced form).

    Subclass and implement ``forward``; implement ``backward`` to make the op
    differentiable and ``explain`` to document it. Instances are callable and are
    registered under ``name`` on creation.

    Examples
    --------
    >>> import numpy as np, nvzlab as nvz
    >>> class Square(nvz.Op):
    ...     name = "square"
    ...     category = "custom"
    ...     def forward(self, x):
    ...         return x * x, x
    ...     def backward(self, ctx, grad_out):
    ...         return (2 * ctx * grad_out,)
    ...     def explain(self, attrs, in_shapes, out_shape):
    ...         return {"title": "Square", "formula": "x^2", "text": "Squares each element.",
    ...                 "shape_rule": "same shape"}
    >>> square = Square()
    >>> square(np.array([3.0]))
    array([9.])
    """

    name: str = ""
    category: str = "custom"
    backward_formula: str | None = None
    backward_text: str | None = None

    def __init__(self) -> None:
        if not self.name:
            raise ValueError(f"{type(self).__name__} must define a class attribute `name`.")
        if not hasattr(self, "origin"):
            self.origin = _origin(type(self).__module__)
        registry.register(self)

    # --- API for subclasses -------------------------------------------------------------
    def forward(self, *inputs: np.ndarray, **attrs: Any) -> tuple[np.ndarray, Any]:
        """Compute the output; return ``(out, ctx)`` where ``ctx`` is kept for backward."""
        raise NotImplementedError

    def backward(self, ctx: Any, grad_out: np.ndarray) -> tuple[np.ndarray | None, ...]:
        """Return one gradient per input (``None`` for inputs that get no gradient)."""
        raise NotImplementedError

    def explain(
        self, attrs: dict[str, Any], in_shapes: list[tuple[int, ...]], out_shape: tuple[int, ...]
    ) -> Explanation | None:
        """Return ``{"title", "formula", "text", "shape_rule"}`` or ``None`` (opaque op)."""
        return None

    def node_attrs(self, ctx: Any) -> dict[str, Any]:
        """Extra attrs stored on the recorded node, derived from the backward context.

        Used for data a renderer needs that is neither an input nor an output, such as the
        argmax routing of a max pool. Default: none.
        """
        return {}

    def explain_backward(
        self, attrs: dict[str, Any], in_shapes: list[tuple[int, ...]], out_shape: tuple[int, ...]
    ) -> Explanation | None:
        """Explanation of the backward rule, built from ``backward_formula``/``backward_text``.

        ``None`` when neither is set, so the viewer never shows an invented chain rule.
        """
        if self.backward_formula is None and self.backward_text is None:
            return None
        return {
            "title": f"{self.name} (backward)",
            "formula": self.backward_formula,
            "text": self.backward_text,
            "shape_rule": "each gradient has the shape of the input it belongs to",
        }

    @property
    def differentiable(self) -> bool:
        """Whether a backward rule exists."""
        return type(self).backward is not Op.backward

    # --- calling ----------------------------------------------------------------------
    def __call__(self, *inputs: Any, **attrs: Any) -> np.ndarray:
        arrays = [np.asarray(a, dtype=np.float64) for a in inputs]
        tape = current_tape()
        if tape is None:
            return self.forward(*arrays, **attrs)[0]
        in_ids = [tape.watch(orig, computed=arr) for orig, arr in zip(inputs, arrays, strict=True)]
        out, ctx = self.forward(*arrays, **attrs)
        out = np.asarray(out, dtype=np.float64)
        if any(out is a for a in arrays):
            out = out.copy()  # an op must never alias its input
        tape.record(self, in_ids, out, attrs, ctx)
        return out

    def __repr__(self) -> str:
        tag = "" if self.differentiable else ", non-differentiable"
        return f"<op {self.name} ({self.category}{tag})>"


def _origin(module: str) -> str:
    return "builtin" if module.startswith("nvzlab.") else "user"


class FunctionOp(Op):
    """An op built from a plain function by ``@nvz.op``."""

    def __init__(
        self,
        fn: Callable[..., np.ndarray],
        name: str,
        category: str,
        formula: str | None,
        text: str | None,
        shape_rule: str | None,
        backward_formula: str | None = None,
        backward_text: str | None = None,
    ) -> None:
        self.name = name
        self.category = category
        self.formula = formula
        self.text = text
        self.shape_rule = shape_rule
        self.backward_formula = backward_formula
        self.backward_text = backward_text
        self.origin = _origin(fn.__module__)
        self._fn = fn
        self._bwd: Callable[..., Any] | None = None
        self.__doc__ = fn.__doc__
        self.__name__ = name
        self.__wrapped__ = fn
        super().__init__()

    def forward(self, *inputs: np.ndarray, **attrs: Any) -> tuple[np.ndarray, Any]:
        out = self._fn(*inputs, **attrs)
        return out, (inputs, out, attrs)

    def backward(self, *args: Any) -> Any:
        """Register a backward rule (``@op.backward``) or evaluate it (``op.backward(ctx, g)``)."""
        if len(args) == 1 and callable(args[0]):
            self._bwd = args[0]
            return args[0]
        if self._bwd is None:
            raise NotImplementedError(f"op {self.name!r} has no backward rule")
        (ctx, grad_out) = args
        inputs, out, attrs = ctx
        grads = self._bwd(grad_out, *inputs, out, **attrs)
        if not isinstance(grads, tuple):
            grads = (grads,)
        if len(grads) != len(inputs):
            raise ValueError(
                f"backward of op {self.name!r} returned {len(grads)} gradient(s) for "
                f"{len(inputs)} input(s). Fix: return one gradient per input, in order."
            )
        return grads

    @property
    def differentiable(self) -> bool:
        return self._bwd is not None

    def explain(self, attrs, in_shapes, out_shape):
        if self.formula is None and self.text is None:
            return None
        return {
            "title": self.name,
            "formula": self.formula,
            "text": self.text,
            "shape_rule": self.shape_rule,
        }

    @property
    def __signature__(self) -> inspect.Signature:
        return inspect.signature(self._fn)


def op(
    fn: Callable[..., np.ndarray] | None = None,
    *,
    name: str | None = None,
    category: str = "custom",
    formula: str | None = None,
    text: str | None = None,
    shape_rule: str | None = None,
    backward_formula: str | None = None,
    backward_text: str | None = None,
) -> Any:
    """Turn a plain NumPy function into a recordable op.

    Without a ``.backward`` rule the op is non-differentiable: the backward pass stops at
    it and the trace says so. Without ``formula`` and ``text`` the op is opaque: it is
    rendered with its name, shapes and values only. ``backward_formula`` and
    ``backward_text`` describe the backward rule in terms of the upstream gradient ``g``;
    without them backward nodes carry no explanation.

    Examples
    --------
    >>> import numpy as np, nvzlab as nvz
    >>> @nvz.op(category="activation", formula=r"x \\cdot \\sigma(x)", text="Smooth gate.")
    ... def swish(x):
    ...     return x / (1.0 + np.exp(-x))
    >>> @swish.backward
    ... def _(grad, x, out):
    ...     s = 1.0 / (1.0 + np.exp(-x))
    ...     return grad * (s + x * s * (1.0 - s))
    >>> swish(np.array([0.0]))
    array([0.])
    """

    def wrap(f: Callable[..., np.ndarray]) -> FunctionOp:
        return FunctionOp(
            f,
            name or f.__name__,
            category,
            formula,
            text,
            shape_rule,
            backward_formula,
            backward_text,
        )

    return wrap(fn) if fn is not None else wrap
