"""Loss ops: bce, cross_entropy."""

from __future__ import annotations

import numpy as np

from nvzlab.core.errors import check_shape
from nvzlab.core.op import op

EPS = 1e-12


@op(
    category="loss",
    formula=r"L = -\frac{1}{n}\sum_i \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]",
    text=(
        "Binary cross-entropy: how surprised the model is by the true labels, averaged "
        "over samples. Probabilities are clipped to [1e-12, 1 - 1e-12] before the log for "
        "stability."
    ),
    shape_rule="p and y share a shape -> scalar",
    backward_formula=r"\frac{\partial L}{\partial p} = g \cdot \frac{p - y}{n\,p(1-p)}",
    backward_text=(
        "The loss starts the backward pass, so g is 1. Each probability's gradient points "
        "away from its label and grows as p approaches the wrong extreme; labels get no "
        "gradient."
    ),
)
def bce(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean binary cross-entropy between probabilities ``p`` and labels ``y`` (0 or 1).

    Examples
    --------
    >>> from nvzlab import ops
    >>> float(ops.bce(np.array([0.5]), np.array([1.0])))
    0.6931471805599453
    """
    check_shape(
        p.shape == y.shape,
        "bce",
        "p and y to have the same shape",
        f"p {p.shape} and y {y.shape}",
        f"reshape y to {p.shape} (one label per prediction)",
    )
    pc = np.clip(p, EPS, 1.0 - EPS)
    return np.asarray(-np.mean(y * np.log(pc) + (1.0 - y) * np.log1p(-pc)))


@bce.backward
def _(grad, p, y, out):
    inside = (p > EPS) & (p < 1.0 - EPS)
    pc = np.clip(p, EPS, 1.0 - EPS)
    gp = grad * inside * (-(y / pc) + (1.0 - y) / (1.0 - pc)) / p.size
    return gp, None


@op(
    category="loss",
    formula=r"L = -\frac{1}{n}\sum_i \log \mathrm{softmax}(z_i)_{y_i}",
    text=(
        "Cross-entropy from raw scores (logits): applies softmax and takes the negative "
        "log probability of the true class, averaged over samples. Computed with "
        "log-sum-exp for stability."
    ),
    shape_rule="logits (n, C) and labels (n,) class indices or (n, C) one-hot -> scalar",
    backward_formula=(
        r"\frac{\partial L}{\partial z} = g \cdot "
        r"\frac{\mathrm{softmax}(z) - \mathrm{onehot}(y)}{n}"
    ),
    backward_text=(
        "Softmax and cross-entropy cancel into a simple rule: predicted probabilities minus "
        "the one-hot labels, divided by the number of samples. Labels get no gradient."
    ),
)
def cross_entropy(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean softmax cross-entropy from logits.

    Examples
    --------
    >>> from nvzlab import ops
    >>> float(ops.cross_entropy(np.array([[0.0, 0.0]]), np.array([1])))
    0.6931471805599453
    """
    check_shape(
        logits.ndim == 2,
        "cross_entropy",
        "logits to be 2-D (n_samples, n_classes)",
        f"logits {logits.shape}",
        "reshape a single sample as logits.reshape(1, -1)",
    )
    onehot = _onehot(y, logits.shape)
    lse = np.log(np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1)) + logits.max(
        axis=1
    )
    return np.asarray(np.mean(lse - (logits * onehot).sum(axis=1)))


@cross_entropy.backward
def _(grad, logits, y, out):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    sm = e / e.sum(axis=1, keepdims=True)
    return grad * (sm - _onehot(y, logits.shape)) / logits.shape[0], None


def _onehot(y: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    n, c = shape
    if y.shape == (n, c):
        return y
    check_shape(
        y.shape == (n,),
        "cross_entropy",
        f"labels with shape ({n},) (class indices) or ({n}, {c}) (one-hot) for logits {shape}",
        f"y {y.shape}",
        "pass one integer class index per sample",
    )
    idx = y.astype(int)
    check_shape(
        bool(np.all((idx >= 0) & (idx < c))) and bool(np.all(idx == y)),
        "cross_entropy",
        f"integer class indices in [0, {c - 1}]",
        f"labels {y.tolist()}",
        "use class indices, not probabilities",
    )
    return np.eye(c)[idx]
