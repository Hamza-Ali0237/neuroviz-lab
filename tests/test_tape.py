import numpy as np
import pytest

import nvzlab as nvz
from conftest import validate
from nvzlab import ops
from nvzlab.core.errors import NvzError


def test_explicit_tape_backward_and_trace():
    x, w = np.array([[1.0, 2.0]]), np.array([0.5, -0.5])
    with nvz.Tape(name="demo") as t:
        t.watch(w, role="param", name="w")
        loss = ops.sum(ops.matmul(x, w))
        grads = t.backward(loss)
    np.testing.assert_allclose(grads["w"], [1.0, 2.0])
    tr = t.to_trace()
    assert tr.kind == "train" and tr.model["name"] == "demo" and tr.model["provenance"] == "user"
    assert [n.phase for n in tr.nodes] == ["forward", "forward", "backward", "backward"]
    validate(tr)


def test_steps_and_events_and_auto_steps():
    with nvz.Tape() as t:
        with nvz.step("A"):
            a = ops.relu(np.array([1.0]))
            nvz.event("note", {"k": 1}, refs=[a])
        ops.relu(a)  # outside any step -> automatic step named after the op
    assert [s.title for s in t.steps] == ["A", "relu"]
    assert t.steps[0].nodes == ["n0"] and t.steps[0].events == ["e0"] and t.steps[1].nodes == ["n1"]
    assert t.events[0].refs == [t.tensor_id(a)]


def test_step_and_event_are_noops_outside_tape():
    with nvz.step("nothing"):
        nvz.event("x", {})


def test_no_nested_tapes():
    with nvz.Tape(), pytest.raises(NvzError, match="already active"):
        with nvz.Tape():
            pass


def test_identity_tracking_reuses_tensor_ids():
    w = np.array([0.5, -0.5])
    with nvz.Tape() as t:
        ops.matmul(np.array([[1.0, 2.0]]), w)
        ops.matmul(np.array([[3.0, 4.0]]), w)
    assert t.nodes[0].inputs[1] == t.nodes[1].inputs[1]
    assert len(t.tensors) == 5


def test_user_op_registration_and_backward():
    @nvz.op(name="test_square", category="custom", formula="x^2", text="Squares.")
    def square(x):
        return x * x

    @square.backward
    def _(grad, x, out):
        return 2 * x * grad

    assert "test_square" in nvz.list_ops() and square.differentiable
    assert square(np.array([3.0])).tolist() == [9.0]
    with nvz.Tape() as t:
        loss = ops.sum(square(np.array([3.0])))
        grads = t.backward(loss)
    np.testing.assert_allclose(grads["t0"], [6.0])


def test_opaque_and_non_differentiable_op():
    @nvz.op
    def test_opaque(x):
        return np.abs(x)

    assert not test_opaque.differentiable
    w = np.array([1.0, -2.0])
    with nvz.Tape() as t:
        t.watch(w, role="param", name="w")
        loss = ops.sum(test_opaque(w))
        grads = t.backward(loss)
    node = t.nodes[0]
    assert node.explain is None and node.differentiable is False
    assert "w" not in grads
    stops = [e for e in t.events if e.kind == "backward_stopped"]
    assert len(stops) == 1 and stops[0].payload["op"] == "test_opaque"


def test_op_class_form():
    class Cube(nvz.Op):
        name = "test_cube"

        def forward(self, x):
            return x**3, x

        def backward(self, ctx, grad_out):
            return (3 * ctx**2 * grad_out,)

        def explain(self, attrs, in_shapes, out_shape):
            return {"title": "Cube", "formula": "x^3", "text": "Cubes.", "shape_rule": "same"}

    cube = Cube()
    assert cube.differentiable and nvz.core.registry.get("test_cube") is cube
    with nvz.Tape() as t:
        loss = ops.sum(cube(np.array([2.0])))
        grads = t.backward(loss)
    np.testing.assert_allclose(grads["t0"], [12.0])
    assert t.nodes[0].explain["title"] == "Cube"


def test_cannot_overwrite_builtin_op():
    with pytest.raises(NvzError, match="built-in op"):

        @nvz.op
        def sigmoid(x):
            return x


def test_user_op_redefinition_is_allowed():
    @nvz.op
    def test_redef(x):
        return x

    @nvz.op  # noqa: F811 (re-running a notebook cell)
    def test_redef(x):  # noqa: F811
        return x * 2

    assert nvz.core.registry.get("test_redef")(np.array([1.0])).tolist() == [2.0]


def test_backward_rule_returning_wrong_count_is_reported():
    @nvz.op
    def test_bad(a, b):
        return a + b

    @test_bad.backward
    def _(grad, a, b, out):
        return grad

    with nvz.Tape() as t, pytest.raises(ValueError, match="returned 1 gradient"):
        t.backward(ops.sum(test_bad(np.ones(2), np.ones(2))))


def test_backward_needs_recorded_loss():
    with nvz.Tape() as t, pytest.raises(NvzError, match="output of a recorded op"):
        t.backward(np.array(1.0))


def test_sgd_update_matches_formula():
    w, b = np.array([1.0, 2.0]), np.array([0.5])
    x, y = np.array([[1.0, -1.0]]), np.array([1.0])
    with nvz.Tape() as t:
        t.watch(w, role="param", name="w")
        t.watch(b, role="param", name="b")
        loss = ops.bce(ops.sigmoid(ops.linear(x, w, b)), y)
        grads = t.backward(loss)
        new = t.sgd_update({"w": w, "b": b}, grads, lr=0.3)
    np.testing.assert_allclose(new["w"], w - 0.3 * grads["w"])
    np.testing.assert_allclose(new["b"], b - 0.3 * grads["b"])
    assert w.tolist() == [1.0, 2.0]
    ev = [e for e in t.events if e.kind == "sgd_update"]
    assert [e.payload["param"] for e in ev] == ["w", "b"] and ev[0].payload["lr"] == 0.3
