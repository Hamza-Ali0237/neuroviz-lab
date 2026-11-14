"""Tiny multi-channel CNN, written with the public level-2 API."""

from __future__ import annotations

from importlib import resources

import numpy as np

from nvzlab import datasets, ops
from nvzlab.core.errors import NvzError
from nvzlab.core.tape import context, event, step
from nvzlab.models.base import Model

PARAM_NAMES = ("W_conv", "b_conv", "W_out", "b_out")


def forward(x: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
    """conv 3->4 (3x3, padding 1) -> ReLU -> max pool 2x2 -> flatten -> linear 64->3 -> softmax."""
    if x.ndim != 4 or x.shape[1:] != (3, 8, 8):
        raise NvzError(
            f"cnn: expected x with shape (n_samples, 3, 8, 8), received {x.shape}. "
            "Fix: pass RGB images of 8x8 pixels, channels first (x.reshape(-1, 3, 8, 8))."
        )
    with step("Input"):
        event("input", {"n_samples": int(x.shape[0]), "image_shape": [3, 8, 8]}, refs=[x])
    with step(
        "Convolve (4 kernels, 3×3×3)",
        backward_title="Gradients reach the kernels",
        layer="Convolution",
    ):
        z = ops.conv2d(x, p["W_conv"], p["b_conv"], stride=1, padding=1)
    with step("ReLU", backward_title="Gradient through ReLU", layer="Convolution"):
        a = ops.relu(z)
    with step(
        "Max pool 2×2", backward_title="Gradient routes through max pooling", layer="Pooling"
    ):
        pooled = ops.maxpool2d(a, size=2, stride=2)
    with step(
        "Flatten", backward_title="Gradient reshaped back to feature maps", layer="Classifier"
    ):
        flat = ops.flatten(pooled)
    with step(
        "Weighted sum (logits)",
        backward_title="Gradients reach W_out and b_out",
        layer="Classifier",
    ):
        logits = ops.linear(flat, p["W_out"], p["b_out"])
    with step("Softmax", backward_title="Gradient through softmax", layer="Classifier"):
        prob = ops.softmax(logits)
    with step("Prediction"):
        event(
            "prediction",
            {
                "predicted_class": prob.argmax(axis=1).tolist(),
                "probability": prob.max(axis=1).tolist(),
            },
            refs=[prob],
        )
    return prob


class CNN(Model):
    """3x8x8 image -> conv 3->4 (3x3) -> ReLU -> max pool -> 64 -> 3 classes, on ``shapes``."""

    name = "cnn"
    display_name = "CNN"
    family = "cnn"
    description = (
        "3x8x8 image -> conv 3->4 (3x3, pad 1) -> ReLU -> max pool 2x2 -> flatten (64) -> "
        "3 classes (softmax); cross-entropy on shapes"
    )
    supports = {"samples": ("single", "batch")}
    loss = "nll"

    def __init__(self) -> None:
        with resources.as_file(resources.files("nvzlab.weights") / "cnn.npz") as path:
            with np.load(path) as z:
                self.params = {k: z[k] for k in PARAM_NAMES}
                self.initial_params = {k: z["initial_" + k] for k in PARAM_NAMES}
        for p in (*self.params.values(), *self.initial_params.values()):
            p.setflags(write=False)
        self.dataset = datasets.load("shapes")

    def forward(self, x: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
        return forward(x, p)

    def record_context(self, x, y, params, kind) -> None:
        ds = self.dataset
        for k, name in enumerate(ds.class_names):
            context(
                f"example_{name}",
                ds.X[np.flatnonzero(ds.y == k)[0]],
                f"One '{name}' image from the shapes dataset (3 channels x 8 x 8 pixels), "
                "for orientation. It is not part of the computation.",
            )
