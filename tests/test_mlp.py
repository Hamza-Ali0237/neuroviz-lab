"""Built-in MLP: parity with a plain NumPy reference and PyTorch, gradient checks, layers."""

import numpy as np
import pytest

import nvzlab as nvz
from conftest import validate

NAMES = ("W1", "b1", "W2", "b2")


def reference(x, p):
    """Forward pass and gradients by hand: 2 -> 4 (ReLU) -> 2 softmax, mean NLL."""
    z1 = x @ p["W1"] + p["b1"]
    h = np.maximum(z1, 0)
    z2 = h @ p["W2"] + p["b2"]
    e = np.exp(z2 - z2.max(axis=1, keepdims=True))
    return z1, h, z2, e / e.sum(axis=1, keepdims=True)


def reference_grads(x, y, p):
    z1, h, z2, prob = reference(x, p)
    n = len(y)
    g2 = (prob - np.eye(2)[y.astype(int)]) / n  # dL/dz2
    g1 = (g2 @ p["W2"].T) * (z1 > 0)  # dL/dz1
    return {
        "W2": h.T @ g2,
        "b2": g2.sum(axis=0),
        "W1": x.T @ g1,
        "b1": g1.sum(axis=0),
        "x": g1 @ p["W1"].T,
    }


def first_values(tr):
    vals = {}
    for t in tr.tensors.values():
        vals.setdefault(t.name, t.array)
    return vals


def test_pretrained_accuracy_on_moons_is_at_least_90_percent():
    m = nvz.models.load("mlp")
    prob = m(m.dataset.X)
    acc = np.mean(prob.argmax(axis=1) == m.dataset.y)
    assert acc >= 0.9, acc
    np.testing.assert_allclose(prob, reference(m.dataset.X, m.params)[3])
    assert all(not v.flags.writeable for v in m.params.values())


def test_train_trace_matches_numpy_reference():
    tr = nvz.trace("mlp", kind="train", samples="batch", lr=0.1)
    v = first_values(tr)
    x, y = v["x"], v["y"]
    p = {k: v[k] for k in NAMES}
    np.testing.assert_array_equal(p["W1"], nvz.models.load("mlp").initial_params["W1"])
    ref = reference_grads(x, y, p)
    for k in (*NAMES, "x"):
        np.testing.assert_allclose(v["grad_" + k], ref[k], rtol=1e-6, atol=1e-9, err_msg=k)
    prob = reference(x, p)[3]
    np.testing.assert_allclose(v["output"], prob)
    np.testing.assert_allclose(v["loss"], -np.mean(np.log(prob[np.arange(len(y)), y.astype(int)])))


def test_numerical_gradient_check_over_every_parameter():
    tr = nvz.trace("mlp", kind="train", samples="batch")
    v = first_values(tr)
    x, y = v["x"], v["y"].astype(int)
    p = {k: v[k] for k in NAMES}

    def loss(q):
        prob = reference(x, q)[3]
        return -np.mean(np.log(prob[np.arange(len(y)), y]))

    eps = 1e-6
    for k in NAMES:
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
    tr = nvz.trace("mlp", kind="train", samples="batch")
    v = first_values(tr)
    x, y = torch.tensor(v["x"]), torch.tensor(v["y"].astype(np.int64))
    p = {k: torch.tensor(v[k], requires_grad=True) for k in NAMES}
    h = torch.relu(x @ p["W1"] + p["b1"])
    prob = torch.softmax(h @ p["W2"] + p["b2"], dim=1)
    loss = torch.nn.functional.nll_loss(torch.log(prob), y)
    loss.backward()
    np.testing.assert_allclose(v["output"], prob.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(v["loss"], loss.item(), atol=1e-6)
    for k in NAMES:
        np.testing.assert_allclose(v["grad_" + k], p[k].grad.numpy(), atol=1e-6, err_msg=k)


def test_layers_and_phase_order_per_iteration():
    tr = nvz.trace("mlp", kind="train", iterations=2)
    validate(tr)
    for it in (0, 1):
        steps = [s for s in tr.steps if s.iteration == it]
        assert [p for p in dict.fromkeys(s.phase for s in steps)] == [
            "forward",
            "loss",
            "backward",
            "update",
        ]
        assert [(s.title, s.layer) for s in steps] == [
            ("Input", None),
            ("Weighted sum", "Hidden layer"),
            ("ReLU", "Hidden layer"),
            ("Weighted sum (logits)", "Output layer"),
            ("Softmax", "Output layer"),
            ("Prediction", None),
            ("Loss", None),
            ("Gradient of the loss", None),
            ("Gradient through softmax", "Output layer"),
            ("Gradients reach W2 and b2", "Output layer"),
            ("Gradient through ReLU", "Hidden layer"),
            ("Gradients reach W1 and b1", "Hidden layer"),
            ("Update W1, b1, W2 and b2", None),
        ]
    updates = [e.payload["param"] for e in tr.events if e.kind == "sgd_update"]
    assert updates == list(NAMES) * 2
    losses = [t.array for t in tr.tensors.values() if t.role == "loss"]
    assert losses[1] < losses[0]
    assert tr.model["architecture"] == [
        {"id": "n0", "label": "Weighted sum", "op": "linear", "params": ["W1", "b1"]},
        {"id": "n1", "label": "ReLU", "op": "relu", "params": []},
        {"id": "n2", "label": "Weighted sum (logits)", "op": "linear", "params": ["W2", "b2"]},
        {"id": "n3", "label": "Softmax", "op": "softmax", "params": []},
    ]


def test_predict_defaults_and_prediction_event():
    tr = nvz.trace("mlp")
    assert tr.mode == {"samples": "single", "n_samples": 1, "n_features": 2}
    assert tr.dataset["name"] == "moons" and tr.model["provenance"] == "builtin-pretrained"
    out = next(t for t in tr.tensors.values() if t.role == "output")
    pred = next(e for e in tr.events if e.kind == "prediction")
    assert pred.payload["predicted_class"] == out.array.argmax(axis=1).tolist()
    np.testing.assert_allclose(out.array.sum(axis=1), 1.0)


def test_moons_is_not_linearly_separable_and_read_only():
    from sklearn.linear_model import LogisticRegression

    ds = nvz.datasets.load("moons")
    assert ds.X.shape == (120, 2) and sorted(set(ds.y)) == [0, 1]
    assert ds.feature_names == ["x1", "x2"] and ds.class_names == ["0", "1"]
    assert LogisticRegression(C=1e6, max_iter=5000).fit(ds.X, ds.y).score(ds.X, ds.y) < 1.0
    with pytest.raises(ValueError):
        ds.X[0, 0] = 0.0
    np.testing.assert_array_equal(nvz.datasets.load("moons").X, ds.X)


def test_prob_grid_matches_trace_predictions_at_three_points():
    for kind in ("predict", "train"):
        tr = nvz.trace("mlp", kind=kind, samples="batch", iterations=2)
        ctx = {t.name: t.array for t in tr.tensors.values() if t.role == "context"}
        n_grids = 1 if kind == "predict" else 3  # before training, after update 1, after update 2
        assert [f"prob_grid_{k}" in ctx for k in range(n_grids)] == [True] * n_grids
        assert f"prob_grid_{n_grids}" not in ctx
        assert ctx["prob_grid_0"].shape == (41, 41)
        g1, g2 = ctx["prob_grid_x1"], ctx["prob_grid_x2"]
        ds = nvz.models.load("mlp").dataset
        np.testing.assert_allclose(
            [g1[0], g1[-1]],
            ds.X[:, 0].min() - 0.5 + [0, 0] + [0, ds.X[:, 0].max() - ds.X[:, 0].min() + 1.0],
        )
        for k in range(n_grids):
            # the params of grid k: the first-seen params (k = 0) or the `after` of update k
            if k == 0:
                params: dict = {}
                for t in tr.tensors.values():
                    if t.role == "param":
                        params.setdefault(t.name, t.array)
            else:
                ups = [e for e in tr.events if e.kind == "sgd_update"]
                params = {
                    e.payload["param"]: tr.tensors[e.payload["after"]].array
                    for e in ups[(k - 1) * 4 : k * 4]
                }
            assert set(params) == {"W1", "b1", "W2", "b2"}
            for i, j in [(0, 0), (20, 20), (40, 7)]:
                pt = np.array([[g1[i], g2[j]]])
                direct = nvz.trace("mlp", pt, params=params).tensors
                prob = next(t.array for t in direct.values() if t.role == "output")
                assert ctx[f"prob_grid_{k}"][i, j] == pytest.approx(float(prob[0, 1]), rel=1e-9)
        assert "moons data range plus 0.5" in next(
            t.note for t in tr.tensors.values() if t.name == "prob_grid_0"
        )
