import numpy as np
import pytest

import nvzlab as nvz
from conftest import gradcheck
from nvzlab import ops
from nvzlab.core.errors import ShapeError


def test_matmul(rng):
    a, b = rng.normal(size=(3, 4)), rng.normal(size=(4, 2))
    np.testing.assert_allclose(ops.matmul(a, b), a @ b)
    gradcheck(ops.matmul, [a, b])
    gradcheck(ops.matmul, [a, rng.normal(size=4)])  # 1-D right operand


def test_add_broadcast(rng):
    a, b = rng.normal(size=(3, 4)), rng.normal(size=4)
    np.testing.assert_allclose(ops.add(a, b), a + b)
    gradcheck(ops.add, [a, b])
    gradcheck(ops.add, [a, rng.normal(size=(3, 1))])
    gradcheck(ops.add, [rng.normal(size=(1, 4)), rng.normal(size=(3, 1))])
    gradcheck(ops.add, [a, np.array(2.0)])


def test_linear(rng):
    x, w, b = rng.normal(size=(5, 3)), rng.normal(size=(3, 2)), rng.normal(size=2)
    np.testing.assert_allclose(ops.linear(x, w, b), x @ w + b)
    gradcheck(ops.linear, [x, w, b])
    gradcheck(ops.linear, [x, rng.normal(size=3), rng.normal(size=1)])


def test_sigmoid(rng):
    x = rng.normal(size=(2, 3)) * 3
    np.testing.assert_allclose(ops.sigmoid(x), 1 / (1 + np.exp(-x)))
    assert ops.sigmoid(np.array([1000.0, -1000.0])).tolist() == [1.0, 0.0]  # no overflow
    gradcheck(ops.sigmoid, [x])


def test_relu(rng):
    x = rng.normal(size=(4, 3))
    np.testing.assert_allclose(ops.relu(x), np.maximum(x, 0))
    gradcheck(ops.relu, [x])


def test_softmax(rng):
    x = rng.normal(size=(3, 5)) * 4
    out = ops.softmax(x)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    np.testing.assert_allclose(ops.softmax(np.array([1000.0, 0.0])), [1.0, 0.0])
    gradcheck(ops.softmax, [x])
    gradcheck(ops.softmax, [x], axis=0)
    with pytest.raises(ShapeError, match="softmax: expected axis"):
        ops.softmax(x, axis=2)


def test_bce(rng):
    p, y = rng.uniform(0.05, 0.95, size=6), rng.integers(0, 2, size=6).astype(float)
    expected = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    np.testing.assert_allclose(ops.bce(p, y), expected)
    assert np.isfinite(ops.bce(np.array([0.0, 1.0]), np.array([1.0, 0.0])))  # clipped, finite
    gradcheck(ops.bce, [p, y], diff=[0])
    assert ops.bce.backward(ops.bce.forward(p, y)[1], np.array(1.0))[1] is None


def test_cross_entropy(rng):
    z, y = rng.normal(size=(4, 3)) * 3, np.array([0, 2, 1, 2])
    sm = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    expected = -np.mean(np.log(sm[np.arange(4), y]))
    np.testing.assert_allclose(ops.cross_entropy(z, y), expected)
    np.testing.assert_allclose(ops.cross_entropy(z, np.eye(3)[y]), expected)  # one-hot
    assert np.isfinite(ops.cross_entropy(np.array([[1000.0, 0.0]]), np.array([1])))
    gradcheck(ops.cross_entropy, [z, y], diff=[0])
    gradcheck(ops.cross_entropy, [z, np.eye(3)[y]], diff=[0])


def test_mean_and_sum(rng):
    x = rng.normal(size=(3, 4))
    for fn, ref in ((ops.mean, np.mean), (ops.sum, np.sum)):
        for axis in (None, 0, 1, -1):
            np.testing.assert_allclose(fn(x, axis=axis), ref(x, axis=axis))
            gradcheck(fn, [x], axis=axis)
        gradcheck(fn, [x])


def test_every_builtin_op_is_documented():
    for name in nvz.list_ops():
        o = nvz.core.registry.get(name)
        if o.origin != "builtin":
            continue
        e = o.explain({}, [], ())
        assert e and e["formula"] and e["text"] and e["shape_rule"], name
        assert o.differentiable == (name != "tree_predict"), name


def test_ops_run_as_plain_numpy_outside_tape():
    out = ops.sigmoid(ops.linear(np.array([[1.0, 2.0]]), np.array([0.5, -0.5]), np.array([0.0])))
    assert isinstance(out, np.ndarray) and out.dtype == np.float64
    assert nvz.core.tape.current_tape() is None


def test_tanh(rng):
    x = rng.normal(size=(2, 3)) * 3
    np.testing.assert_allclose(ops.tanh(x), np.tanh(x))
    gradcheck(ops.tanh, [x])


def test_mul_broadcast(rng):
    a, b = rng.normal(size=(3, 4)), rng.normal(size=4)
    np.testing.assert_allclose(ops.mul(a, b), a * b)
    gradcheck(ops.mul, [a, b])
    gradcheck(ops.mul, [a, rng.normal(size=(3, 1))])
    gradcheck(ops.mul, [rng.normal(size=(1, 4)), rng.normal(size=(3, 1))])
    gradcheck(ops.mul, [a, np.array(2.0)])
    with pytest.raises(ShapeError, match="mul: expected shapes that broadcast"):
        ops.mul(a, rng.normal(size=3))


def test_nll(rng):
    p = rng.uniform(0.05, 0.95, size=(5, 3))
    p /= p.sum(axis=1, keepdims=True)
    y = rng.integers(0, 3, size=5)
    np.testing.assert_allclose(ops.nll(p, y), -np.mean(np.log(p[np.arange(5), y])))
    np.testing.assert_allclose(ops.nll(p, np.eye(3)[y]), ops.nll(p, y))  # one-hot labels
    # softmax then nll equals cross_entropy from logits, forward and backward
    z = rng.normal(size=(5, 3))
    np.testing.assert_allclose(ops.nll(ops.softmax(z), y), ops.cross_entropy(z, y))
    gradcheck(ops.nll, [p, y.astype(float)], diff=[0])
    assert float(ops.nll(np.array([[0.0, 1.0]]), np.array([0]))) == -np.log(1e-12)  # clipped
    with pytest.raises(ShapeError, match="nll: expected p to be 2-D"):
        ops.nll(np.array([0.5, 0.5]), np.array([0]))


def test_tree_predict():
    # root: x1 <= 0.5 -> node 1 (counts 4,1) else node 2 (1,4)
    tree = dict(
        feature=np.array([0, -1, -1]),
        threshold=np.array([0.5, 0.0, 0.0]),
        left=np.array([1, -1, -1]),
        right=np.array([2, -1, -1]),
        counts=np.array([[5, 5], [4, 1], [1, 4]]),
    )
    x = np.array([[0.2, 9.0], [0.5, 9.0], [0.9, 9.0]])
    out = ops.tree_predict(x, *tree.values())
    np.testing.assert_allclose(out, [[0.8, 0.2], [0.8, 0.2], [0.2, 0.8]])  # <= goes left
    assert not ops.tree_predict.differentiable
    with nvz.Tape() as t:
        loss = ops.sum(ops.tree_predict(x, *tree.values()))
        grads = t.backward(loss)
    assert [e.kind for e in t.events] == ["backward_stopped"]
    assert t.tensor_id(x) not in grads and not any(n.op == "tree_predict" for n in t.nodes[1:])
    with pytest.raises(ShapeError, match="tree_predict: expected feature indices below"):
        ops.tree_predict(x[:, :1], [1, -1, -1], *list(tree.values())[1:])
    with pytest.raises(ShapeError, match="tree_predict: expected counts"):
        ops.tree_predict(x, tree["feature"], tree["threshold"], tree["left"], tree["right"], [1])
