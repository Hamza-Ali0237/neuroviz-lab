"""Ops used by attention models: embedding, layernorm."""

from __future__ import annotations

import numpy as np

from nvzlab.core.errors import check_shape
from nvzlab.core.op import op


@op(
    category="embedding",
    formula=r"E_{t} = \mathrm{table}[\,\mathrm{id}_t\,]",
    text=(
        "Looks up one row of the table per token id, so every token becomes a learned "
        "vector. The ids themselves are not changed by training; the rows are."
    ),
    shape_rule="ids (n, T) and table (V, d) -> (n, T, d)",
    backward_formula=(
        r"\frac{\partial L}{\partial \mathrm{table}[v]} = \sum_{t:\ \mathrm{id}_t = v} g_t"
    ),
    backward_text=(
        "Each token's gradient is added to the row it was looked up from; a row used twice "
        "receives both. The ids are integers and get no gradient."
    ),
)
def embedding(ids: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Row lookup ``table[ids]`` for integer-valued ``ids``.

    Examples
    --------
    >>> from nvzlab import ops
    >>> ops.embedding(np.array([[1, 0]]), np.array([[0.0, 0.0], [1.0, 2.0]]))
    array([[[1., 2.],
            [0., 0.]]])
    """
    check_shape(
        table.ndim == 2,
        "embedding",
        "table to be 2-D (vocabulary, d_model)",
        f"table {table.shape}",
        "reshape the table to (V, d)",
    )
    idx = ids.astype(int)
    check_shape(
        bool(np.all(idx == ids)) and bool(np.all((idx >= 0) & (idx < table.shape[0]))),
        "embedding",
        f"integer token ids in [0, {table.shape[0] - 1}]",
        f"ids with values in [{ids.min()}, {ids.max()}]" if ids.size else "empty ids",
        f"use ids from 0 to {table.shape[0] - 1} (the table has {table.shape[0]} rows)",
    )
    return table[idx]


@embedding.backward
def _(grad, ids, table, out):
    g = np.zeros_like(table)
    np.add.at(g, ids.astype(int).ravel(), grad.reshape(-1, table.shape[1]))
    return None, g


@op(
    category="norm",
    formula=(
        r"\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}},"
        r"\quad y = \gamma \odot \hat{x} + \beta"
    ),
    text=(
        "Centres and rescales each vector (the last axis) to mean 0 and variance 1, then "
        "lets two learned vectors scale and shift it again. Keeps activations in a steady "
        "range layer after layer."
    ),
    shape_rule="x (..., d), gamma (d,), beta (d,) -> (..., d); mean and variance over d",
    backward_formula=(
        r"\frac{\partial L}{\partial x} = \frac{1}{\sqrt{\sigma^2+\epsilon}}\left(g' - "
        r"\overline{g'} - \hat{x}\,\overline{g' \odot \hat{x}}\right),\ g' = g \odot \gamma;"
        r"\quad \frac{\partial L}{\partial \gamma} = \sum g \odot \hat{x},\ "
        r"\frac{\partial L}{\partial \beta} = \sum g"
    ),
    backward_text=(
        "Because every output of a vector depends on its mean and variance, the gradient "
        "to x is the scaled upstream gradient with its mean removed and its component along "
        "the normalised vector removed, divided by the standard deviation. Gamma collects "
        "g times the normalised values, beta collects g."
    ),
)
def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalisation over the last axis.

    Examples
    --------
    >>> from nvzlab import ops
    >>> ops.layernorm(np.array([[1.0, 3.0]]), np.ones(2), np.zeros(2)).round(4)
    array([[-1., 1.]])
    """
    d = x.shape[-1] if x.ndim else 0
    check_shape(
        x.ndim >= 1 and gamma.shape == (d,) and beta.shape == (d,),
        "layernorm",
        f"gamma and beta with shape ({d},) to match the last axis of x {x.shape}",
        f"gamma {gamma.shape} and beta {beta.shape}",
        f"use one scale and one shift per feature, shape ({d},)",
    )
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * ((x - mu) / np.sqrt(var + eps)) + beta


@layernorm.backward
def _(grad, x, gamma, beta, out, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(x.var(axis=-1, keepdims=True) + eps)
    xhat = (x - mu) * inv
    gp = grad * gamma
    gx = inv * (
        gp - gp.mean(axis=-1, keepdims=True) - xhat * (gp * xhat).mean(axis=-1, keepdims=True)
    )
    lead = tuple(range(grad.ndim - 1))
    return gx, (grad * xhat).sum(axis=lead), grad.sum(axis=lead)
