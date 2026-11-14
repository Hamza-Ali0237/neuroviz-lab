"""Decision tree classifier: a non-gradient, event-based algorithm on the same tape.

The tree is grown level by level with ``nvz.step``/``nvz.event`` (train traces) and
stored as plain arrays; prediction follows each sample from the root to a leaf (predict
traces) and ends in the ``tree_predict`` op.
"""

from __future__ import annotations

from collections.abc import Iterator
from importlib import resources
from typing import Any

import numpy as np

from nvzlab import datasets, ops
from nvzlab.core.errors import NvzError
from nvzlab.core.tape import context, event, step
from nvzlab.models.base import Model

MAX_DEPTH = 3
CRITERIA = ("gini", "entropy")
ARRAYS = ("feature", "threshold", "left", "right", "counts")
EXPLAIN = {
    "gini": {
        "title": "Gini impurity",
        "formula": r"G = 1 - \sum_k p_k^2",
        "text": (
            "How often a random sample of the node would be mislabelled if labelled by the "
            "node's class proportions; 0 means every sample has the same class."
        ),
    },
    "entropy": {
        "title": "Entropy",
        "formula": r"H = -\sum_k p_k \log_2 p_k",
        "text": (
            "How many bits are needed on average to name a sample's class given the node's "
            "class proportions; 0 means every sample has the same class."
        ),
    },
}


def impurity(counts: np.ndarray, criterion: str) -> float:
    """Gini impurity or entropy (base 2) of a vector of class counts."""
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    if criterion == "gini":
        return float(1.0 - np.sum(p * p))
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


class Growth:
    """Grows a tree on ``(x, y)`` one depth level at a time, recording events.

    Thresholds are midpoints between sorted unique values; ties are broken by the lowest
    feature index, then the lowest threshold. Node ids are assigned in creation order.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        criterion: str,
        max_depth: int = MAX_DEPTH,
        n_classes: int = 2,
    ) -> None:
        self.x, self.y = x, y.astype(int)
        self.refs = [x, y]  # the arrays as the caller holds them, for event refs
        self.names, self.criterion, self.max_depth = feature_names, criterion, max_depth
        self.n_classes = max(n_classes, int(self.y.max()) + 1)
        self.nodes: list[dict[str, Any]] = []
        self._add(np.arange(len(self.y)), 0, None, None)

    # --- growing -------------------------------------------------------------------
    def level(self, depth: int) -> None:
        """Decide every pending node at ``depth``: split it or make it a leaf."""
        for node in [n for n in self.nodes if n["depth"] == depth and n["pending"]]:
            self._decide(node)

    def finish(self) -> None:
        """Turn the remaining pending nodes into leaves and record the whole tree."""
        pending = [n for n in self.nodes if n["pending"]]
        if pending:
            with step("Growth stops", layer=f"Depth {pending[0]['depth']}"):
                for node in pending:
                    reason = (
                        "maximum depth reached"
                        if node["depth"] >= self.max_depth
                        else f"growing stopped after depth {node['depth'] - 1}"
                    )
                    self._leaf(node, reason)
        with step("Grown tree"):
            event("tree_structure", self.structure(), refs=self.refs)

    def _decide(self, node: dict[str, Any]) -> None:
        nid, depth = node["id"], node["depth"]
        reason = None
        if node["impurity"] == 0.0:
            reason = "pure: every sample has the same class"
        elif node["n_samples"] < 2:
            reason = "a single sample cannot be split"
        elif depth >= self.max_depth:
            reason = "maximum depth reached"
        if reason is not None:
            with step(f"Node {nid} becomes a leaf", layer=f"Depth {depth}"):
                self._leaf(node, reason)
            return
        with step(f"Split node {nid}", layer=f"Depth {depth}"):
            idx = node["idx"]
            cands = self._candidates(idx, node["impurity"])
            event(
                "split_search",
                {
                    "node": nid,
                    "depth": depth,
                    "n_samples": node["n_samples"],
                    "counts": node["counts"],
                    "impurity": node["impurity"],
                    "criterion": self.criterion,
                    "explain": EXPLAIN[self.criterion],
                    "candidates": cands,
                },
                refs=self.refs,
            )
            if not cands:
                self._leaf(node, "no distinct values to split on")
                return
            best = int(np.argmax([c["gain"] for c in cands]))  # first max: ties -> lowest
            c = cands[best]
            node.update(pending=False, feature=c["feature"], threshold=c["threshold"])
            event("split_chosen", {"node": nid, "candidate": best, **c})
            go_left = self.x[idx, c["feature"]] <= c["threshold"]
            for side, child_idx in (("left", idx[go_left]), ("right", idx[~go_left])):
                child = self._add(child_idx, depth + 1, nid, side)
                node[side] = child["id"]
                event(
                    "node_created",
                    {k: child[k] for k in ("id", "depth", "parent", "side", "counts")}
                    | {"n_samples": child["n_samples"], "impurity": child["impurity"]},
                )

    def _candidates(self, idx: np.ndarray, parent_impurity: float) -> list[dict[str, Any]]:
        y, n = self.y[idx], len(idx)
        out = []
        for j in range(self.x.shape[1]):
            values = np.unique(self.x[idx, j])
            for t in (values[:-1] + values[1:]) / 2.0:
                left = y[self.x[idx, j] <= t]
                right = y[self.x[idx, j] > t]
                lc = np.bincount(left, minlength=self.n_classes)
                rc = np.bincount(right, minlength=self.n_classes)
                li, ri = impurity(lc, self.criterion), impurity(rc, self.criterion)
                weighted = (len(left) * li + len(right) * ri) / n
                out.append(
                    {
                        "feature": j,
                        "feature_name": self.names[j],
                        "threshold": float(t),
                        "left_counts": lc.tolist(),
                        "right_counts": rc.tolist(),
                        "left_impurity": li,
                        "right_impurity": ri,
                        "weighted_impurity": weighted,
                        "gain": parent_impurity - weighted,
                    }
                )
        return out

    def _add(self, idx: np.ndarray, depth: int, parent: int | None, side: str | None) -> dict:
        counts = np.bincount(self.y[idx], minlength=self.n_classes)
        node = {
            "id": len(self.nodes),
            "depth": depth,
            "parent": parent,
            "side": side,
            "idx": idx,
            "n_samples": int(len(idx)),
            "counts": counts.tolist(),
            "impurity": impurity(counts, self.criterion),
            "feature": -1,
            "threshold": 0.0,
            "left": -1,
            "right": -1,
            "pending": True,
            "reason": None,
        }
        self.nodes.append(node)
        return node

    def _leaf(self, node: dict[str, Any], reason: str) -> None:
        node.update(pending=False, reason=reason)
        event("leaf", {"node": node["id"], "reason": reason, **leaf_payload(node["counts"])})

    # --- export --------------------------------------------------------------------
    def arrays(self) -> dict[str, np.ndarray]:
        """The tree as arrays (``feature``, ``threshold``, ``left``, ``right``, ``counts``)."""
        return {
            "feature": np.array([n["feature"] for n in self.nodes]),
            "threshold": np.array([n["threshold"] for n in self.nodes]),
            "left": np.array([n["left"] for n in self.nodes]),
            "right": np.array([n["right"] for n in self.nodes]),
            "counts": np.array([n["counts"] for n in self.nodes]),
        }

    def structure(self) -> dict[str, Any]:
        return structure(self.arrays(), self.names, self.criterion, self.max_depth)


def grow(
    x: np.ndarray, y: np.ndarray, feature_names: list[str], criterion: str, levels: int = MAX_DEPTH
) -> dict[str, np.ndarray]:
    """Grow a tree ``levels`` deep and return it as arrays (events recorded if a tape is active)."""
    g = Growth(x, y, feature_names, criterion)
    for depth in range(levels):
        g.level(depth)
    g.finish()
    return g.arrays()


def leaf_payload(counts: list[int]) -> dict[str, Any]:
    total = sum(counts)
    predicted = int(np.argmax(counts))
    return {
        "counts": list(counts),
        "predicted_class": predicted,
        "probability": counts[predicted] / total if total else 0.0,
    }


def structure(
    tree: dict[str, np.ndarray], feature_names: list[str], criterion: str, max_depth: int
) -> dict[str, Any]:
    """JSON description of a tree stored as arrays: every node with its depth and children."""
    feature, left, right = (tree[k].astype(int) for k in ("feature", "left", "right"))
    depth = np.zeros(len(feature), dtype=int)
    for i in range(len(feature)):
        for child in (left[i], right[i]):
            if child >= 0:
                depth[child] = depth[i] + 1
    nodes = []
    for i in range(len(feature)):
        counts = [int(c) for c in tree["counts"][i]]
        leaf = feature[i] < 0
        nodes.append(
            {
                "id": i,
                "depth": int(depth[i]),
                "is_leaf": bool(leaf),
                "feature": None if leaf else int(feature[i]),
                "feature_name": None if leaf else feature_names[feature[i]],
                "threshold": None if leaf else float(tree["threshold"][i]),
                "left": None if leaf else int(left[i]),
                "right": None if leaf else int(right[i]),
                "n_samples": sum(counts),
                "impurity": impurity(np.array(counts), criterion),
                **leaf_payload(counts),
            }
        )
    return {
        "criterion": criterion,
        "max_depth": max_depth,
        "features": list(feature_names),
        "n_nodes": len(nodes),
        "nodes": nodes,
    }


class DecisionTree(Model):
    """Classification tree (depth 3) on ``blobs`` or ``moons``; Gini or entropy splits."""

    name = "decision_tree"
    display_name = "Decision Tree"
    family = "tree"
    description = "classification tree, depth 3, midpoint thresholds; Gini or entropy"
    supports = {
        "samples": ("single", "batch"),
        "features": (1, 2),
        "dataset": ("blobs", "moons"),
        "criterion": CRITERIA,
    }
    initial_params: dict[str, np.ndarray] = {}
    max_depth = MAX_DEPTH

    def __init__(
        self, features: int | None = None, dataset: str = "blobs", criterion: str = "gini"
    ) -> None:
        self.check_option("dataset", dataset)
        self.check_option("criterion", criterion)
        if features is None:
            features = 2
        self.check_option("features", features)
        if dataset == "moons" and features != 2:
            raise NvzError(
                f"decision_tree: dataset='moons' has 2 features, received features={features}. "
                "Fix: drop `features`, or use dataset='blobs' for a 1-feature tree."
            )
        self.features, self.criterion = features, criterion
        self.dataset = datasets.load(dataset, features=features)
        fname = f"decision_tree_{dataset}_f{features}_{criterion}.npz"
        with resources.as_file(resources.files("nvzlab.weights") / fname) as path:
            with np.load(path) as z:
                self.params = {k: z[k] for k in ARRAYS}
        for p in self.params.values():
            p.setflags(write=False)

    def record_context(
        self, x: np.ndarray, y: np.ndarray | None, p: dict[str, np.ndarray], kind: str
    ) -> None:
        ds = self.dataset
        what = "the data the tree is grown on" if kind == "train" else "for orientation"
        context(
            "dataset_X",
            ds.X,
            f"The whole {ds.name} dataset, {what}. Feature-space plots draw the leaf regions "
            "over it.",
        )
        context("dataset_y", ds.y, f"Labels of the whole {ds.name} dataset (0 or 1).")

    def forward(self, x: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.features:
            raise NvzError(
                f"decision_tree: expected x with shape (n_samples, {self.features}), "
                f"received {x.shape}. "
                f"Fix: reshape x, or load the model with features={x.shape[-1]}."
            )
        names = self.dataset.feature_names
        tree = structure(p, names, self.criterion, self.max_depth)
        nodes = tree["nodes"]
        with step("Tree"):
            event("tree_structure", tree, refs=[p[k] for k in ARRAYS])
        at = [0] * len(x)  # current node of each sample
        done = [False] * len(x)
        depth = 0
        while not all(done):
            with step(f"Depth {depth}"):
                for i in range(len(x)):
                    if done[i]:
                        continue
                    node = nodes[at[i]]
                    if node["is_leaf"]:
                        event(
                            "leaf",
                            {"sample": i, "node": node["id"], **leaf_payload(node["counts"])},
                        )
                        done[i] = True
                        continue
                    value = float(x[i, node["feature"]])
                    go_left = value <= node["threshold"]
                    at[i] = node["left"] if go_left else node["right"]
                    event(
                        "node_test",
                        {
                            "sample": i,
                            "node": node["id"],
                            "feature": node["feature"],
                            "feature_name": node["feature_name"],
                            "threshold": node["threshold"],
                            "value": value,
                            "direction": "left" if go_left else "right",
                            "next_node": at[i],
                        },
                    )
            depth += 1
        with step("Prediction"):
            prob = ops.tree_predict(x, *(p[k] for k in ARRAYS))
            event(
                "prediction",
                {
                    "predicted_class": prob.argmax(axis=1).tolist(),
                    "probability": prob.max(axis=1).tolist(),
                },
                refs=[prob],
            )
        return prob

    def train(self, x: np.ndarray, y: np.ndarray, iterations: int) -> Iterator[int]:
        """Grow the tree ``iterations`` levels deep on ``(x, y)``, one iteration per level."""
        g = Growth(x, y, self.dataset.feature_names, self.criterion, self.max_depth)
        for depth in range(iterations):
            yield depth
            g.level(depth)
        g.finish()
