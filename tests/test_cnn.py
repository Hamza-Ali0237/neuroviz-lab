"""Built-in CNN: shapes dataset, parity with PyTorch, gradient checks, layers, pool routing."""

import numpy as np
import pytest

import nvzlab as nvz
from conftest import validate
from nvzlab import ops
from nvzlab.models.cnn import PARAM_NAMES, forward


def first_values(tr):
    vals = {}
    for t in tr.tensors.values():
        vals.setdefault(t.name, t.array)
    return vals


def test_shapes_dataset_is_small_balanced_and_read_only():
    ds = nvz.datasets.load("shapes")
    assert ds.X.shape == (90, 3, 8, 8) and ds.y.shape == (90,)
    assert ds.X.min() >= 0.0 and ds.X.max() <= 1.0
    assert np.bincount(ds.y).tolist() == [30, 30, 30]
    assert ds.class_names == ["square", "ring", "cross"] and ds.feature_names == ["R", "G", "B"]
    with pytest.raises(ValueError):
        ds.X[0, 0, 0, 0] = 1.0
    np.testing.assert_array_equal(nvz.datasets.load("shapes").X, ds.X)
    x, y = ds.sample(4)
    assert x.shape == (4, 3, 8, 8) and not x.flags.writeable


def test_pretrained_accuracy_on_shapes_is_at_least_90_percent():
    m = nvz.models.load("cnn")
    prob = m(m.dataset.X)
    assert prob.shape == (90, 3)
    np.testing.assert_allclose(prob.sum(axis=1), 1.0)
    assert np.mean(prob.argmax(axis=1) == m.dataset.y) >= 0.9
    assert all(not v.flags.writeable for v in m.params.values())
    with pytest.raises(
        nvz.core.errors.NvzError, match=r"expected x with shape \(n_samples, 3, 8, 8\)"
    ):
        m(np.zeros((1, 8, 8)))


def test_numerical_gradient_check_over_every_parameter():
    tr = nvz.trace("cnn", kind="train", samples="batch")
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
    tr = nvz.trace("cnn", kind="train", samples="batch")
    v = first_values(tr)
    x = torch.tensor(v["x"], requires_grad=True)
    y = torch.tensor(v["y"].astype(np.int64))
    p = {k: torch.tensor(v[k], requires_grad=True) for k in PARAM_NAMES}
    h = fn.max_pool2d(torch.relu(fn.conv2d(x, p["W_conv"], p["b_conv"], padding=1)), 2)
    prob = torch.softmax(h.flatten(1) @ p["W_out"] + p["b_out"], dim=1)
    loss = fn.nll_loss(torch.log(prob), y)
    loss.backward()
    np.testing.assert_allclose(v["output"], prob.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(v["loss"], loss.item(), atol=1e-6)
    for k in (*PARAM_NAMES, "x"):
        g = (x if k == "x" else p[k]).grad.numpy()
        np.testing.assert_allclose(v["grad_" + k], g, atol=1e-6, err_msg=k)
        print(f"parity ok: grad_{k} {g.shape} max|diff|={np.abs(v['grad_' + k] - g).max():.2e}")


def test_max_pool_routing_is_recorded_and_gradient_flows_only_to_winners():
    tr = nvz.trace("cnn", kind="train", samples="batch")
    fwd = next(n for n in tr.nodes if n.op == "maxpool2d" and n.phase == "forward")
    bwd = next(n for n in tr.nodes if n.op == "maxpool2d" and n.phase == "backward")
    assert bwd.forward_of == fwd.id
    argmax = np.array(fwd.attrs["argmax"])
    pooled_in = tr.tensors[fwd.inputs[0]].array
    pooled_out = tr.tensors[fwd.outputs[0]].array
    assert argmax.shape == pooled_out.shape == (4, 4, 4, 4)
    flat = pooled_in.reshape(4, 4, 64)
    np.testing.assert_array_equal(
        np.take_along_axis(flat, argmax.reshape(4, 4, 16), axis=-1).reshape(4, 4, 4, 4),
        pooled_out,
    )
    g_in = tr.tensors[bwd.outputs[0]].array.reshape(4, 4, 64)
    g_out = tr.tensors[bwd.inputs[0]].array
    winners = np.zeros_like(g_in, dtype=bool)
    np.put_along_axis(winners, argmax.reshape(4, 4, 16), True, axis=-1)
    assert np.all(g_in[~winners] == 0.0)
    np.testing.assert_allclose(
        np.take_along_axis(g_in, argmax.reshape(4, 4, 16), axis=-1).reshape(4, 4, 4, 4), g_out
    )
    assert "argmax" in bwd.explain["text"]


def test_layers_backward_order_and_context_examples():
    tr = nvz.trace("cnn", kind="train")
    validate(tr)
    assert [(s.title, s.layer, s.phase) for s in tr.steps] == [
        ("Input", None, "forward"),
        ("Convolve (4 kernels, 3×3×3)", "Convolution", "forward"),
        ("ReLU", "Convolution", "forward"),
        ("Max pool 2×2", "Pooling", "forward"),
        ("Flatten", "Classifier", "forward"),
        ("Weighted sum (logits)", "Classifier", "forward"),
        ("Softmax", "Classifier", "forward"),
        ("Prediction", None, "forward"),
        ("Loss", None, "loss"),
        ("Gradient of the loss", None, "backward"),
        ("Gradient through softmax", "Classifier", "backward"),
        ("Gradients reach W_out and b_out", "Classifier", "backward"),
        ("Gradient reshaped back to feature maps", "Classifier", "backward"),
        ("Gradient routes through max pooling", "Pooling", "backward"),
        ("Gradient through ReLU", "Convolution", "backward"),
        ("Gradients reach the kernels", "Convolution", "backward"),
        ("Update W_conv, b_conv, W_out and b_out", None, "update"),
    ]
    ctx = {t.name: t for t in tr.tensors.values() if t.role == "context"}
    assert ctx.pop("loss_path").shape == (2,)  # every train trace records the exact losses
    assert sorted(ctx) == ["example_cross", "example_ring", "example_square"]
    assert all(t.shape == (3, 8, 8) and "orientation" in t.note for t in ctx.values())
    assert [a["op"] for a in tr.model["architecture"]] == [
        "conv2d",
        "relu",
        "maxpool2d",
        "flatten",
        "linear",
        "softmax",
    ]
    assert tr.model["architecture"][0]["params"] == ["W_conv", "b_conv"]
    np.testing.assert_array_equal(
        first_values(tr)["W_conv"], nvz.models.load("cnn").initial_params["W_conv"]
    )


def test_loss_decreases_over_iterations():
    tr = nvz.trace("cnn", kind="train", samples="batch", iterations=2)
    losses = [float(t.array) for t in tr.tensors.values() if t.role == "loss"]
    assert len(losses) == 2 and losses[1] < losses[0]
    assert tr.mode == {"samples": "batch", "n_samples": 4, "n_features": 192}
