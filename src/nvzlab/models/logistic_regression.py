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
MIN_HALF_WIDTH = 0.15


class LogisticRegression(Model):
    """Binary classifier: ``p = sigmoid(x @ w + b)``, trained on ``blobs``."""

    name = "logistic_regression"
    display_name = "Logistic Regression"
    family = "linear"
    supports = {"samples": ("single", "batch"), "features": (1, 2)}
    loss = "bce"
    default_lr = 0.5  # three default iterations travel visibly downhill (see tests)

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

    def record_path(
        self,
        x: np.ndarray,
        y: np.ndarray,
        path: list[dict[str, np.ndarray]],
        losses: list[float],
    ) -> None:
        """Loss landscape: mean BCE on a grid around the training path (two parameters vary,
        the rest stay at their initial values). The window is centred on the path with a
        half-width of max(1.5 × path extent, 0.15) per axis, so the path spans the window."""
        p0 = path[0]
        if self.features == 2:
            names = ("w1", "w2")
            coords = np.array([[float(p["w"][0]), float(p["w"][1])] for p in path])
            fixed = f"b fixed at its initial value {float(p0['b'][0]):.4g}"
        else:
            names = ("w", "b")
            coords = np.array([[float(p["w"][0]), float(p["b"][0])] for p in path])
            fixed = "nothing else is held fixed (w and b are the only parameters)"
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        centre, half = (lo + hi) / 2, np.maximum(1.5 * (hi - lo), MIN_HALF_WIDTH)
        grid = [np.linspace(c - h, c + h, LOSS_GRID) for c, h in zip(centre, half, strict=True)]
        a0, a1 = np.meshgrid(grid[0], grid[1], indexing="ij")
        if self.features == 2:
            z = x[:, 0, None, None] * a0 + x[:, 1, None, None] * a1 + p0["b"][0]
        else:
            z = x[:, 0, None, None] * a0 + a1
        pc = np.clip(1.0 / (1.0 + np.exp(-z)), EPS, 1.0 - EPS)
        yy = y[:, None, None]
        loss = -np.mean(yy * np.log(pc) + (1.0 - yy) * np.log1p(-pc), axis=0)
        n = x.shape[0]
        window = ", ".join(
            f"{nm} in [{c - h:.4g}, {c + h:.4g}]"
            for nm, c, h in zip(names, centre, half, strict=True)
        )
        context(
            "loss_landscape",
            loss,
            f"Mean binary cross-entropy over the {n} sample(s) of this trace on a "
            f"{LOSS_GRID}×{LOSS_GRID} grid of ({names[0]}, {names[1]}) with {fixed}. Window "
            f"centred on the training path, half-width max(1.5 × path extent, {MIN_HALF_WIDTH}): "
            f"{window}. Real computed values; path and marker heights are the exact losses "
            f"in loss_path, not grid interpolations.",
        )
        for name, g in zip(names, grid, strict=True):
            context(f"loss_landscape_{name}", g, f"Grid values of {name} for loss_landscape.")
