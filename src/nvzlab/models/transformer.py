"""Mini Transformer encoder (one block, one head), written with the public level-2 API."""

from __future__ import annotations

from importlib import resources

import numpy as np

from nvzlab import datasets, ops
from nvzlab.core.errors import NvzError
from nvzlab.core.tape import event, name, step
from nvzlab.models.base import Model

D_MODEL = 8
D_FF = 16
SEQ_LEN = 6
PARAM_NAMES = (
    "E_tok",
    "E_pos",
    "ln1_gamma",
    "ln1_beta",
    "W_q",
    "W_k",
    "W_v",
    "W_o",
    "ln2_gamma",
    "ln2_beta",
    "W_ff1",
    "b_ff1",
    "W_ff2",
    "b_ff2",
    "W_out",
    "b_out",
)


def forward(x: np.ndarray, p: dict[str, np.ndarray], vocab: list[str] | None = None) -> np.ndarray:
    """Embeddings -> pre-norm block (attention, feed-forward, residuals) -> mean -> 2 classes."""
    if x.ndim != 2 or x.shape[1] != SEQ_LEN:
        raise NvzError(
            f"transformer: expected x with shape (n_samples, {SEQ_LEN}) of token ids, "
            f"received {x.shape}. Fix: pass {SEQ_LEN} integer ids per sequence."
        )
    with step("Input"):
        payload = {"n_samples": int(x.shape[0]), "seq_len": SEQ_LEN}
        if vocab is not None:
            payload["vocab"] = vocab
            payload["tokens"] = [[vocab[int(i)] for i in row] for row in x]
        event("input", payload, refs=[x])

    with step(
        "Token embedding", backward_title="Gradients reach the token table", layer="Embedding"
    ):
        e = ops.embedding(x, p["E_tok"])
    with step(
        "Add positional embedding",
        backward_title="Gradients reach the position table",
        layer="Embedding",
    ):
        h0 = ops.add(e, p["E_pos"])

    with step("Layer norm", backward_title="Gradient through layer norm", layer="Attention"):
        xn = ops.layernorm(h0, p["ln1_gamma"], p["ln1_beta"])
    with step("Queries Q = XW_q", backward_title="Gradients reach W_q", layer="Attention"):
        q = ops.matmul(xn, p["W_q"])
    with step("Keys K = XW_k", backward_title="Gradients reach W_k", layer="Attention"):
        k = ops.matmul(xn, p["W_k"])
    with step("Values V = XW_v", backward_title="Gradients reach W_v", layer="Attention"):
        v = ops.matmul(xn, p["W_v"])
    with step("Scores QKᵀ", backward_title="Gradient splits into Q and K", layer="Attention"):
        scores = ops.matmul(q, ops.transpose(k, axes=(0, 2, 1)))
    with step("Scale by 1/√d", backward_title="Gradient scaled by 1/√d", layer="Attention"):
        scaled = ops.scale(scores, c=1.0 / np.sqrt(D_MODEL))
    with step(
        "Attention weights (softmax per row)",
        backward_title="Gradient through the softmax rows",
        layer="Attention",
    ):
        a = name(ops.softmax(scaled, axis=-1), "attention weights")
    with step(
        "Weighted sum of values A·V",
        backward_title="Gradient splits into A and V",
        layer="Attention",
    ):
        mixed = ops.matmul(a, v)
    with step("Output projection", backward_title="Gradients reach W_o", layer="Attention"):
        attn = ops.matmul(mixed, p["W_o"])

    with step(
        "Residual add (attention)",
        backward_title="Gradient flows to both residual paths",
        layer="Residual",
    ):
        h1 = ops.add(h0, attn)

    with step("Layer norm", backward_title="Gradient through layer norm", layer="Feed-forward"):
        xn2 = ops.layernorm(h1, p["ln2_gamma"], p["ln2_beta"])
    with step(
        "Expand 8 → 16", backward_title="Gradients reach W_ff1 and b_ff1", layer="Feed-forward"
    ):
        f1 = ops.linear(xn2, p["W_ff1"], p["b_ff1"])
    with step("ReLU", backward_title="Gradient through ReLU", layer="Feed-forward"):
        f1 = ops.relu(f1)
    with step(
        "Project 16 → 8", backward_title="Gradients reach W_ff2 and b_ff2", layer="Feed-forward"
    ):
        f2 = ops.linear(f1, p["W_ff2"], p["b_ff2"])

    with step(
        "Residual add (feed-forward)",
        backward_title="Gradient flows to both residual paths",
        layer="Residual",
    ):
        h2 = ops.add(h1, f2)

    with step(
        "Mean over positions", backward_title="Gradient shared by all positions", layer="Classifier"
    ):
        pooled = ops.mean(h2, axis=1)
    with step(
        "Weighted sum (logits)",
        backward_title="Gradients reach W_out and b_out",
        layer="Classifier",
    ):
        logits = ops.linear(pooled, p["W_out"], p["b_out"])
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


class Transformer(Model):
    """One encoder block (single-head attention, feed-forward, residuals) on ``tokens``."""

    name = "transformer"
    family = "transformer"
    description = (
        "6 tokens -> embeddings (d=8) -> 1 encoder block (1-head self-attention, feed-forward "
        "8->16->8, residuals) -> mean -> 2 classes (softmax); cross-entropy on tokens"
    )
    supports = {"samples": ("single", "batch"), "heads": (1,)}
    loss = "nll"

    def __init__(self, heads: int = 1) -> None:
        self.check_option("heads", heads)
        self.heads = heads
        with resources.as_file(resources.files("nvzlab.weights") / "transformer.npz") as path:
            with np.load(path) as z:
                self.params = {k: z[k] for k in PARAM_NAMES}
                self.initial_params = {k: z["initial_" + k] for k in PARAM_NAMES}
        for p in (*self.params.values(), *self.initial_params.values()):
            p.setflags(write=False)
        self.dataset = datasets.load("tokens")

    def forward(self, x: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
        return forward(x, p, self.dataset.vocab)
