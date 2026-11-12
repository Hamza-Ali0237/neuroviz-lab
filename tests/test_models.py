import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SkLR

import nvzlab as nvz
from nvzlab import ops


@pytest.mark.parametrize("features", [1, 2])
def test_parity_with_sklearn(features):
    m = nvz.models.load("logistic_regression", features=features)
    sk = SkLR()
    sk.coef_ = m.params["w"].reshape(1, -1)
    sk.intercept_ = m.params["b"]
    sk.classes_ = np.array([0, 1])
    ours = m(m.dataset.X)
    np.testing.assert_allclose(ours, sk.predict_proba(m.dataset.X)[:, 1], atol=1e-6)
    assert np.mean((ours > 0.5) == m.dataset.y) > 0.9


def test_train_gradients_match_hand_derived():
    tr = nvz.trace("logistic_regression", kind="train", samples="batch", lr=0.1)
    vals = {}
    for t in tr.tensors.values():  # first tensor per name = the values before the update
        vals.setdefault(t.name, t.array)
    x, y, w, b = vals["x"], vals["y"], vals["w"], vals["b"]
    p = 1 / (1 + np.exp(-(x @ w + b)))
    n = len(y)
    # dL/dz = (p - y) / n ; dL/dw = x^T dL/dz ; dL/db = sum dL/dz ; dL/dx = outer(dL/dz, w)
    dz = (p - y) / n
    np.testing.assert_allclose(vals["grad_w"], x.T @ dz, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(vals["grad_b"], [dz.sum()], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(vals["grad_x"], np.outer(dz, w), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(vals["loss"], -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def test_two_iterations_have_all_phases_and_updates_follow_sgd():
    lr = 0.25
    tr = nvz.trace("logistic_regression", kind="train", iterations=2, lr=lr)
    assert tr.meta["iterations"] == 2 and tr.meta["lr"] == lr
    for it in (0, 1):
        phases = [s.phase for s in tr.steps if s.iteration == it]
        assert [p for p in dict.fromkeys(phases)] == ["forward", "loss", "backward", "update"]
    updates = [e for e in tr.events if e.kind == "sgd_update"]
    assert [e.payload["param"] for e in updates] == ["w", "b", "w", "b"]
    for e in updates:
        before, grad, after = (tr.tensors[e.payload[k]].array for k in ("before", "grad", "after"))
        np.testing.assert_allclose(after, before - lr * grad)
        assert e.refs == [e.payload["before"], e.payload["grad"], e.payload["after"]]
    # iteration 2 starts from the updated params
    second_forward = next(
        n for n in tr.nodes if n.op == "linear" and tr.steps[n.step].iteration == 1
    )
    assert second_forward.inputs[1] == updates[0].payload["after"]
    assert second_forward.inputs[2] == updates[1].payload["after"]
    # the loss goes down
    losses = [t.array for t in tr.tensors.values() if t.role == "loss"]
    assert len(losses) == 2 and losses[1] < losses[0]


def test_train_uses_initial_params_and_never_mutates_them():
    m = nvz.models.load("logistic_regression")
    w0 = m.initial_params["w"].copy()
    tr = nvz.trace("logistic_regression", kind="train")
    first_w = tr.tensors[tr.nodes[0].inputs[1]].array
    np.testing.assert_array_equal(first_w, w0)
    np.testing.assert_array_equal(m.initial_params["w"], w0)
    assert not m.params["w"].flags.writeable


def test_user_params_unchanged_after_train_trace():
    params = {"w": np.array([0.5, -0.3]), "b": np.zeros(1)}
    snapshot = {k: v.copy() for k, v in params.items()}

    def model(x, p):
        return ops.sigmoid(ops.linear(x, p["w"], p["b"]))

    x, y = np.array([[1.0, 2.0], [-1.0, 0.5]]), np.array([1.0, 0.0])
    tr = nvz.trace(model, x, y, params=params, kind="train", loss="bce", iterations=3)
    for k in params:
        np.testing.assert_array_equal(params[k], snapshot[k])
    assert tr.model["provenance"] == "user" and tr.model["name"] == "model"
    assert tr.model["architecture"] == [
        {"id": "n0", "label": "linear", "op": "linear", "params": ["w", "b"]},
        {"id": "n1", "label": "sigmoid", "op": "sigmoid", "params": []},
    ]


def test_model_subclass_and_custom_loss_callable():
    class Tiny(nvz.Model):
        name = "tiny"

        def __init__(self):
            self.params = {"w": np.array([[0.5], [-0.5]]), "b": np.zeros(1)}

        def forward(self, x, p):
            return ops.linear(x, p["w"], p["b"])

    x, y = np.array([[1.0, 2.0]]), np.array([[0.0]])
    tr = nvz.trace(
        Tiny(), x, y, kind="train", loss=lambda out, y: ops.mean(ops.relu(ops.add(out, y)))
    )
    assert tr.kind == "train" and tr.model["name"] == "tiny"
    assert [n.op for n in tr.nodes if n.phase == "loss"] == ["add", "relu", "mean"]


def test_describe_and_lists():
    assert nvz.list_models() == ["logistic_regression"]
    assert nvz.list_datasets() == ["blobs"]
    assert {"linear", "sigmoid", "bce", "softmax"} <= set(nvz.list_ops())
    d = nvz.models.load("logistic_regression", features=1).describe()
    assert "param w: shape (1,)" in d and "loss: bce" in d and "features: 1, 2" in d


def test_dataset_is_read_only_and_sampling_is_deterministic():
    ds = nvz.datasets.load("blobs", features=2)
    assert ds.X.shape == (120, 2) and ds.y.shape == (120,) and set(ds.y) == {0, 1}
    assert ds.feature_names == ["x1", "x2"] and ds.class_names == ["0", "1"]
    with pytest.raises(ValueError):
        ds.X[0, 0] = 1.0
    x1, _ = ds.sample(4)
    x2, _ = ds.sample(4, seed=0)
    np.testing.assert_array_equal(x1, x2)
    assert not x1.flags.writeable
    np.testing.assert_array_equal(nvz.datasets.load("blobs").X, ds.X)


def test_predict_default_is_single_sample_two_features():
    tr = nvz.trace("logistic_regression")
    assert tr.mode == {"samples": "single", "n_samples": 1, "n_features": 2}
    assert tr.tensors["t0"].name == "x" and tr.tensors["t0"].role == "input"
    pred = next(e for e in tr.events if e.kind == "prediction")
    out = next(t for t in tr.tensors.values() if t.role == "output")
    assert pred.payload["predicted_class"] == (out.array > 0.5).astype(int).tolist()
