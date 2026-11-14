"""Built-in mini Transformer: tokens dataset, PyTorch parity, gradient checks, layers, residuals."""

import numpy as np
import pytest

import nvzlab as nvz
from conftest import validate
from nvzlab import ops
from nvzlab.core.errors import NvzError
from nvzlab.models.transformer import PARAM_NAMES, forward


def first_values(tr):
    vals = {}
    for t in tr.tensors.values():
        vals.setdefault(t.name, t.array)
    return vals


def test_tokens_dataset_is_balanced_and_labels_mean_first_equals_last():
    ds = nvz.datasets.load("tokens")
    assert ds.X.shape == (120, 6) and ds.y.shape == (120,)
    assert ds.vocab == list("ABCDEFGH") and ds.class_names == ["different", "same"]
    assert ds.feature_names == [f"pos{i}" for i in range(1, 7)]
    assert np.bincount(ds.y).tolist() == [60, 60]
    assert np.array_equal(ds.X == ds.X.astype(int), np.ones_like(ds.X, dtype=bool))
    assert ds.X.min() == 0 and ds.X.max() == 7
    np.testing.assert_array_equal((ds.X[:, 0] == ds.X[:, -1]).astype(int), ds.y)
    with pytest.raises(ValueError):
        ds.X[0, 0] = 1.0
    np.testing.assert_array_equal(nvz.datasets.load("tokens").X, ds.X)


def test_pretrained_accuracy_on_tokens_is_at_least_90_percent():
    m = nvz.models.load("transformer")
    prob = m(m.dataset.X)
    assert prob.shape == (120, 2)
    np.testing.assert_allclose(prob.sum(axis=1), 1.0)
    assert np.mean(prob.argmax(axis=1) == m.dataset.y) >= 0.9
    assert all(not v.flags.writeable for v in m.params.values())
    with pytest.raises(NvzError, match=r"transformer: expected x with shape \(n_samples, 6\)"):
        m(np.zeros((1, 5)))
    with pytest.raises(NvzError, match="transformer: heads=2 is not supported. Valid values: 1"):
        nvz.trace("transformer", heads=2)


def test_attention_weights_are_a_named_tensor_with_rows_summing_to_one():
    tr = nvz.trace("transformer", samples="batch")
    aw = [t for t in tr.tensors.values() if t.name == "attention weights"]
    assert len(aw) == 1 and aw[0].shape == (4, 6, 6) and aw[0].role == "activation"
    np.testing.assert_allclose(aw[0].array.sum(axis=-1), 1.0)
    node = next(n for n in tr.nodes if aw[0].id in n.outputs)
    assert node.op == "softmax" and node.attrs == {"axis": -1}
    assert tr.steps[node.step].title == "Attention weights (softmax per row)"
    inp = next(e for e in tr.events if e.kind == "input").payload
    assert inp["vocab"] == list("ABCDEFGH") and len(inp["tokens"]) == 4
    x = tr.tensors["t0"].array
    assert inp["tokens"][0] == [inp["vocab"][int(i)] for i in x[0]]


def test_numerical_gradient_check_over_every_parameter():
    tr = nvz.trace("transformer", kind="train", samples="batch")
    v = first_values(tr)
    x, y = v["x"], v["y"]
    p = {k: v[k] for k in PARAM_NAMES}

    def loss(q):
        return float(ops.nll(forward(x, q), y))

    eps = 1e-6
    for k in PARAM_NAMES:
        num = np.zeros_like(p[k])
        for idx in np.ndindex(*p[k].shape):
            plus = {a: b.copy() for a, b in p.items()}
            minus = {a: b.copy() for a, b in p.items()}
            plus[k][idx] += eps
            minus[k][idx] -= eps
            num[idx] = (loss(plus) - loss(minus)) / (2 * eps)
        np.testing.assert_allclose(v["grad_" + k], num, rtol=1e-5, atol=1e-7, err_msg=k)


def test_parity_with_pytorch():
    torch = pytest.importorskip("torch")
    fn = torch.nn.functional
    tr = nvz.trace("transformer", kind="train", samples="batch")
    v = first_values(tr)
    ids = torch.tensor(v["x"].astype(np.int64))
    y = torch.tensor(v["y"].astype(np.int64))
    p = {k: torch.tensor(v[k], requires_grad=True) for k in PARAM_NAMES}
    h0 = p["E_tok"][ids] + p["E_pos"]
    xn = fn.layer_norm(h0, (8,), p["ln1_gamma"], p["ln1_beta"], eps=1e-5)
    q, k, val = xn @ p["W_q"], xn @ p["W_k"], xn @ p["W_v"]
    a = torch.softmax(q @ k.transpose(1, 2) / 8**0.5, dim=-1)
    h1 = h0 + (a @ val) @ p["W_o"]
    xn2 = fn.layer_norm(h1, (8,), p["ln2_gamma"], p["ln2_beta"], eps=1e-5)
    h2 = h1 + torch.relu(xn2 @ p["W_ff1"] + p["b_ff1"]) @ p["W_ff2"] + p["b_ff2"]
    prob = torch.softmax(h2.mean(dim=1) @ p["W_out"] + p["b_out"], dim=1)
    loss = fn.nll_loss(torch.log(prob), y)
    loss.backward()
    aw = next(t for t in tr.tensors.values() if t.name == "attention weights")
    np.testing.assert_allclose(aw.array, a.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(v["output"], prob.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(v["loss"], loss.item(), atol=1e-6)
    for k in PARAM_NAMES:
        g = p[k].grad.numpy()
        np.testing.assert_allclose(v["grad_" + k], g, atol=1e-6, err_msg=k)
        print(f"parity ok: grad_{k} {g.shape} max|diff|={np.abs(v['grad_' + k] - g).max():.2e}")


def test_residual_adds_produce_grad_accumulate_nodes():
    tr = nvz.trace("transformer", kind="train")
    validate(tr)
    acc = [n for n in tr.nodes if n.op == "grad_accumulate"]
    titles = [tr.steps[n.step].title for n in acc]
    assert titles == [
        "Gradient flows to both residual paths",  # h1: feed-forward layer norm + residual add
        "Gradient through layer norm",  # layer-normed x feeds Q, K and V
        "Gradients reach the position table",  # h0: attention layer norm + residual add
    ]
    assert [len(n.inputs) for n in acc] == [2, 3, 2]
    for n in acc:
        total = sum(tr.tensors[t].array for t in n.inputs)
        np.testing.assert_allclose(tr.tensors[n.outputs[0]].array, total)


def test_layers_and_backward_order():
    tr = nvz.trace("transformer", kind="train")
    fwd = [(s.title, s.layer) for s in tr.steps if s.phase == "forward"]
    assert fwd == [
        ("Input", None),
        ("Token embedding", "Embedding"),
        ("Add positional embedding", "Embedding"),
        ("Layer norm", "Attention"),
        ("Queries Q = XW_q", "Attention"),
        ("Keys K = XW_k", "Attention"),
        ("Values V = XW_v", "Attention"),
        ("Scores QKᵀ", "Attention"),
        ("Scale by 1/√d", "Attention"),
        ("Attention weights (softmax per row)", "Attention"),
        ("Weighted sum of values A·V", "Attention"),
        ("Output projection", "Attention"),
        ("Residual add (attention)", "Residual"),
        ("Layer norm", "Feed-forward"),
        ("Expand 8 → 16", "Feed-forward"),
        ("ReLU", "Feed-forward"),
        ("Project 16 → 8", "Feed-forward"),
        ("Residual add (feed-forward)", "Residual"),
        ("Mean over positions", "Classifier"),
        ("Weighted sum (logits)", "Classifier"),
        ("Softmax", "Classifier"),
        ("Prediction", None),
    ]
    bwd = [s for s in tr.steps if s.phase == "backward"]
    assert [s.layer for s in bwd][1:] == [s.layer for s in tr.steps if s.phase == "forward"][1:-1][
        ::-1
    ]
    assert [s.title for s in bwd][:3] == [
        "Gradient of the loss",
        "Gradient through softmax",
        "Gradients reach W_out and b_out",
    ]
    assert bwd[-1].title == "Gradients reach the token table"
    arch = tr.model["architecture"]
    assert [a["op"] for a in arch][:4] == ["embedding", "add", "layernorm", "matmul"]
    assert arch[0]["params"] == ["E_tok"] and arch[3]["params"] == ["W_q"]
    updates = [e.payload["param"] for e in tr.events if e.kind == "sgd_update"]
    assert updates == list(PARAM_NAMES)
    assert tr.mode == {"samples": "single", "n_samples": 1, "n_features": 6}


def test_loss_decreases_over_iterations():
    tr = nvz.trace("transformer", kind="train", samples="batch", iterations=2)
    losses = [float(t.array) for t in tr.tensors.values() if t.role == "loss"]
    assert len(losses) == 2 and losses[1] < losses[0]


def test_pretrained_attention_is_soft_enough_to_read_as_a_heat_map():
    tr = nvz.trace("transformer")
    a = next(t.array for t in tr.tensors.values() if t.name == "attention weights")[0]
    scaled = next(tr.tensors[n.outputs[0]].array for n in tr.nodes if n.op == "scale")[0]
    assert a.max(axis=1).max() < 0.8 and a.max(axis=1).min() > 1 / 6
    assert (np.sort(a, axis=1)[:, -2] > 0.1).all()  # every row has a visible second choice
    assert -10 < scaled.min() and scaled.max() < 10


def test_update_step_with_many_params_has_a_short_title_and_the_full_list():
    tr = nvz.trace("transformer", kind="train")
    up = [s for s in tr.steps if s.phase == "update"]
    assert [s.title for s in up] == ["Update all parameters (16 arrays)"]
    events = [e for e in tr.events if e.id in up[0].events]
    assert events[0].payload["params"] == list(PARAM_NAMES)
    assert all("params" not in e.payload for e in events[1:])
