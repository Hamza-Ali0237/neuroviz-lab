"""Public-name safety, op categories, backward explanations, mirrored steps, fan-in."""

import importlib
import json
import pkgutil

import numpy as np

import nvzlab as nvz
from conftest import validate
from nvzlab import ops


def test_submodule_imports_keep_public_callables():
    for m in pkgutil.walk_packages(nvz.__path__, "nvzlab."):
        importlib.import_module(m.name)
    import nvzlab.tracing.serializer  # noqa: F401

    for name in ("trace", "show", "step", "event", "op"):
        assert callable(getattr(nvz, name)), name
    assert nvz.trace is nvz.api.trace
    assert nvz.trace("logistic_regression").kind == "predict"


def test_no_subpackage_shadows_a_public_function():
    subpackages = {m.name.split(".")[-1] for m in pkgutil.iter_modules(nvz.__path__)}
    public = {n for n in nvz.__all__ if callable(getattr(nvz, n))}
    assert not subpackages & public


def test_node_category_and_origin():
    @nvz.op(category="custom", formula="2x", text="Doubles.")
    def test_double(x):
        return 2 * x

    with nvz.Tape() as t:
        ops.sum(ops.sigmoid(test_double(np.ones(2))))
    assert [(n.category, n.origin) for n in t.nodes] == [
        ("custom", "user"),
        ("activation", "builtin"),
        ("reduce", "builtin"),
    ]
    assert {ops.bce.category, ops.linear.category} == {"loss", "linalg"}
    assert ops.sigmoid.origin == "builtin" and test_double.origin == "user"
    validate(t.to_trace())


def test_every_builtin_backward_node_is_explained():
    tr = nvz.trace("logistic_regression", kind="train", samples="batch")
    back = [n for n in tr.nodes if n.phase == "backward"]
    assert back
    for n in back:
        assert n.explain and n.explain["formula"] and n.explain["text"], n.op
    sig = next(n for n in back if n.op == "sigmoid")
    assert sig.explain["title"] == "sigmoid (backward)"
    for name in nvz.list_ops():
        o = nvz.core.registry.get(name)
        if o.origin == "builtin":
            assert o.explain_backward({}, [], ()), name


def test_user_op_without_backward_explanation_keeps_none():
    @nvz.op(formula="x^2", text="Squares.")
    def test_sq(x):
        return x * x

    @test_sq.backward
    def _(grad, x, out):
        return 2 * x * grad

    with nvz.Tape() as t:
        t.backward(ops.sum(test_sq(np.ones(2))))
    assert t.nodes[3].op == "test_sq" and t.nodes[3].explain is None


def test_op_class_backward_explanation():
    class Cube(nvz.Op):
        name = "test_cube2"
        backward_formula = "3x^2 g"
        backward_text = "Chain rule for the cube."

        def forward(self, x):
            return x**3, x

        def backward(self, ctx, grad_out):
            return (3 * ctx**2 * grad_out,)

    cube = Cube()
    with nvz.Tape() as t:
        t.backward(ops.sum(cube(np.array([2.0]))))
    e = next(n for n in t.nodes if n.phase == "backward" and n.op == "test_cube2").explain
    assert e == {
        "title": "test_cube2 (backward)",
        "formula": "3x^2 g",
        "text": "Chain rule for the cube.",
        "shape_rule": "each gradient has the shape of the input it belongs to",
    }


def test_backward_steps_mirror_forward_steps_per_iteration():
    tr = nvz.trace("logistic_regression", kind="train", iterations=2)
    for it in (0, 1):
        titles = [(s.phase, s.title) for s in tr.steps if s.iteration == it]
        assert titles == [
            ("forward", "Input"),
            ("forward", "Weighted sum z = Xw + b"),
            ("forward", "Sigmoid"),
            ("forward", "Prediction"),
            ("loss", "Loss"),
            ("backward", "Gradient of the loss"),
            ("backward", "Gradient through the sigmoid"),
            ("backward", "Gradients reach w and b"),
            ("update", "Update w and b"),
        ]
    back = [s for s in tr.steps if s.phase == "backward"]
    assert all(len(s.nodes) == 1 for s in back)
    for s in back:
        fwd = tr._node(tr._node(s.nodes[0]).forward_of)
        assert tr.steps[fwd.step].iteration == s.iteration
    validate(tr)


def test_default_backward_title():
    def model(x, p):
        with nvz.step("Hidden"):
            return ops.relu(ops.linear(x, p["W"], p["b"]))

    p = {"W": np.array([[0.5], [-0.3]]), "b": np.zeros(1)}
    tr = nvz.trace(model, np.ones((1, 2)), np.ones((1, 1)), params=p, kind="train", loss="bce")
    assert [s.title for s in tr.steps if s.phase == "backward"] == [
        "Gradient of the loss",
        "Backprop through: Hidden",
    ]


def test_residual_fan_in_records_grad_accumulate():
    def model(x, p):
        return ops.add(x, ops.relu(ops.linear(x, p["W"], p["b"])))

    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 2))
    p = {"W": rng.normal(size=(2, 2)), "b": rng.normal(size=2)}

    with nvz.Tape() as t:
        t.watch(x, role="input", name="x")
        loss = ops.sum(model(x, p))
        grads = t.backward(loss)
    acc = [n for n in t.nodes if n.op == "grad_accumulate"]
    assert len(acc) == 1
    (acc,) = acc
    assert acc.phase == "backward" and acc.category == "reduce" and acc.origin == "builtin"
    assert acc.attrs == {"grad_of": [t.tensor_id(x)]}
    assert acc.explain["formula"] == r"\frac{\partial L}{\partial x} = \sum_k g_k"
    partial_ids = {
        o
        for n in t.nodes
        if n.phase == "backward" and n.op in ("add", "linear")
        for o, src in zip(n.outputs, n.attrs["grad_of"], strict=True)
        if src == t.tensor_id(x)
    }
    assert set(acc.inputs) == partial_ids and len(acc.inputs) == 2
    assert grads["x"] is t.tensors[acc.outputs[0]].array
    assert acc.id in t.steps[acc.step].nodes and t.steps[acc.step].phase == "backward"

    eps, num = 1e-6, np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        num[idx] = (np.sum(model(xp, p)) - np.sum(model(xm, p))) / (2 * eps)
    np.testing.assert_allclose(grads["x"], num, rtol=1e-5, atol=1e-7)
    validate(t.to_trace())


def test_trace_json_has_origin_and_backward_explain():
    d = json.loads(nvz.trace("logistic_regression", kind="train").to_json())
    assert {n["origin"] for n in d["nodes"]} == {"builtin"}
    assert {n["category"] for n in d["nodes"]} == {"linalg", "activation", "loss"}
    assert all(n["explain"] for n in d["nodes"] if n["phase"] == "backward")
