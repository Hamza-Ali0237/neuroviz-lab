"""Tree ops: tree_predict."""

from __future__ import annotations

import numpy as np

from nvzlab.core.errors import check_shape
from nvzlab.core.op import op


@op(
    category="tree",
    formula=(
        r"p_k(x) = \frac{c_{\ell(x), k}}{\sum_j c_{\ell(x), j}},"
        r"\quad \ell(x) = \text{leaf reached by } x"
    ),
    text=(
        "Each sample starts at the root and, at every node, goes left when its value of "
        "the node's feature is <= the threshold, right otherwise, until it reaches a leaf. "
        "The leaf's training class counts, normalized, are the predicted probabilities."
    ),
    shape_rule="x (n, f) with a tree of m nodes and C classes -> (n, C)",
)
def tree_predict(
    x: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Class probabilities from a fitted decision tree stored as arrays.

    Node ``i`` tests ``x[:, feature[i]] <= threshold[i]`` and sends samples to
    ``left[i]`` or ``right[i]``; leaves have ``feature[i] == -1``. ``counts[i]`` holds the
    training class counts of node ``i``. Not differentiable: the backward pass stops here.

    Examples
    --------
    >>> from nvzlab import ops
    >>> ops.tree_predict(np.array([[0.2], [0.9]]), [0, -1, -1], [0.5, 0.0, 0.0], [1, -1, -1],
    ...                  [2, -1, -1], [[5, 5], [4, 1], [1, 4]])
    array([[0.8, 0.2],
           [0.2, 0.8]])
    """
    m = feature.shape[0]
    check_shape(
        x.ndim == 2 and feature.shape == threshold.shape == left.shape == right.shape == (m,),
        "tree_predict",
        "x (n, f) and feature, threshold, left, right of one shape (n_nodes,)",
        f"x {x.shape}, feature {feature.shape}, threshold {threshold.shape}, "
        f"left {left.shape}, right {right.shape}",
        "store one entry per tree node in each array",
    )
    check_shape(
        counts.ndim == 2 and counts.shape[0] == m,
        "tree_predict",
        f"counts (n_nodes={m}, n_classes)",
        f"counts {counts.shape}",
        "give counts one row per tree node",
    )
    feat, lft, rgt = feature.astype(int), left.astype(int), right.astype(int)
    check_shape(
        bool(np.all(feat < x.shape[1])),
        "tree_predict",
        f"feature indices below x.shape[1] = {x.shape[1]}",
        f"feature {feat.tolist()}",
        "pass x with every feature the tree tests",
    )
    node = np.zeros(x.shape[0], dtype=int)
    for _ in range(m):
        inner = feat[node] >= 0
        if not inner.any():
            break
        go_left = x[np.arange(x.shape[0]), np.where(inner, feat[node], 0)] <= threshold[node]
        node = np.where(inner, np.where(go_left, lft[node], rgt[node]), node)
    leaf_counts = counts[node]
    return leaf_counts / leaf_counts.sum(axis=1, keepdims=True)
