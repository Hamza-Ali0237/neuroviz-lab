"""``Model``: a forward function plus params, dataset and loss."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvzlab.core.errors import NvzError
from nvzlab.datasets import Dataset


class Model:
    """Base class for models written with ``nvz.ops``.

    Subclasses implement ``forward(x, params)`` and set ``params``. Everything else is
    optional: ``initial_params`` (for train traces), ``dataset``, ``loss``.

    Examples
    --------
    >>> import numpy as np, nvzlab as nvz
    >>> from nvzlab import ops
    >>> class Tiny(nvz.Model):
    ...     name = "tiny"
    ...     def __init__(self):
    ...         self.params = {"w": np.array([0.5, -0.5]), "b": np.zeros(1)}
    ...     def forward(self, x, p):
    ...         return ops.sigmoid(ops.linear(x, p["w"], p["b"]))
    >>> tr = nvz.trace(Tiny(), np.array([[1.0, 2.0]]))
    """

    name: str = "model"
    family: str = "custom"
    supports: dict[str, tuple[Any, ...]] = {"samples": ("single", "batch")}
    params: dict[str, np.ndarray]
    initial_params: dict[str, np.ndarray] | None = None
    dataset: Dataset | None = None
    loss: str | None = None

    def forward(self, x: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: np.ndarray, params: dict[str, np.ndarray] | None = None) -> np.ndarray:
        return self.forward(x, self.params if params is None else params)

    def check_option(self, option: str, value: Any) -> None:
        """Raise a clear error if ``option=value`` is not supported by this model."""
        valid = self.supports.get(option)
        if valid is not None and value not in valid:
            raise NvzError(
                f"{self.name}: {option}={value!r} is not supported. "
                f"Valid values: {', '.join(repr(v) for v in valid)}."
            )

    def describe(self) -> str:
        """A short human-readable description (name, params, dataset, loss, options)."""
        lines = [f"{self.name} ({self.family})"]
        lines += [f"  param {k}: shape {tuple(v.shape)}" for k, v in self.params.items()]
        if self.dataset is not None:
            lines.append(f"  dataset: {self.dataset.name} {self.dataset.X.shape}")
        if self.loss is not None:
            lines.append(f"  loss: {self.loss}")
        for k, v in self.supports.items():
            lines.append(f"  {k}: {', '.join(repr(o) for o in v)}")
        return "\n".join(lines)
