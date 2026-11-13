"""Logistic regression, written with the public level-2 API."""

from __future__ import annotations

from importlib import resources

import numpy as np

from nvzlab import datasets, ops
from nvzlab.core.errors import NvzError
from nvzlab.core.tape import context, event, step
from nvzlab.models.base import Model
from nvzlab.ops.losses import EPS

LOSS_GRID = 41


class LogisticRegression(Model):
    """Binary classifier: ``p = sigmoid(x @ w + b)``, trained on ``blobs``."""

    name = "logistic_regression"
    family = "linear"
    supports = {"samples": ("single", "batch"), "features": (1, 2)}
    loss = "bce"

    def __init__(self, features: int = 2) -> None:
        self.check_option("features", features)
        self.features = features
        with resources.as_file(
            resources.files("nvzlab.weights") / f"logistic_regression_f{features}.npz"
        ) as path:
            with np.load(path) as z:
                self.params = {"w": z["w"], "b": z["b"]}
                self.initial_params = {"w": z["initial_w"], "b": z["initial_b"]}
        for p in (*self.params.values(), *self.initial_params.values()):
            p.setflags(write=False)
        self.dataset = datasets.load("blobs", features=features)

    def forward(self, x: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.features:
            raise NvzError(
                f"logistic_regression: expected x with shape (n_samples, {self.features}), "
                f"received {x.shape}. "
                f"Fix: reshape x, or load the model with features={x.shape[-1]}."
            )
        with step("Input"):
            event("input", {"n_samples": int(x.shape[0]), "n_features": int(x.shape[1])}, refs=[x])
        with step("Weighted sum z = Xw + b", backward_title="Gradients reach w and b"):
            z = ops.linear(x, p["w"], p["b"])
        with step("Sigmoid", backward_title="Gradient through the sigmoid"):
            prob = ops.sigmoid(z)
        with step("Prediction"):
            event(
                "prediction",
                {"threshold": 0.5, "predicted_class": (prob > 0.5).astype(int).tolist()},
                refs=[prob],
            )
        return prob

    def record_context(
        self, x: np.ndarray, y: np.ndarray | None, p: dict[str, np.ndarray], kind: str
    ) -> None:
        ds = self.dataset
        context(
            "dataset_X",
            ds.X,
            "The whole blobs dataset, for orientation. Only the highlighted samples are "
            "computed in this trace.",
        )
        context("dataset_y", ds.y, "Labels of the whole blobs dataset (0 or 1).")
        if kind != "train" or y is None:
            return
        w, b = p["w"], p["b"]
        if self.features == 2:
            axes = (("w1", float(w[0])), ("w2", float(w[1])))
            fixed = f"b fixed at its initial value {float(b[0]):.4g}"
        else:
            axes = (("w", float(w[0])), ("b", float(b[0])))
            fixed = "nothing else is held fixed (w and b are the only parameters)"
        grid = [np.linspace(c - 1.0, c + 1.0, LOSS_GRID) for _, c in axes]
        a0, a1 = np.meshgrid(grid[0], grid[1], indexing="ij")
        if self.features == 2:
            z = x[:, 0, None, None] * a0 + x[:, 1, None, None] * a1 + b[0]
        else:
            z = x[:, 0, None, None] * a0 + a1
        pc = np.clip(1.0 / (1.0 + np.exp(-z)), EPS, 1.0 - EPS)
        yy = y[:, None, None]
        loss = -np.mean(yy * np.log(pc) + (1.0 - yy) * np.log1p(-pc), axis=0)
        n = x.shape[0]
        context(
            "loss_landscape",
            loss,
            f"Mean binary cross-entropy over the {n} sample(s) of this trace, evaluated on a "
            f"{LOSS_GRID}×{LOSS_GRID} grid of ({axes[0][0]}, {axes[1][0]}) with {fixed}. "
            "Real computed values; the training path moves across this surface.",
        )
        for (name, _), g in zip(axes, grid, strict=True):
            context(f"loss_landscape_{name}", g, f"Grid values of {name} for loss_landscape.")
