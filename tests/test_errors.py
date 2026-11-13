import numpy as np
import pytest

import nvzlab as nvz
from nvzlab import ops
from nvzlab.core.errors import NvzError, ShapeError, UnknownNameError


def test_unknown_model_suggests_closest():
    with pytest.raises(
        UnknownNameError,
        match="Unknown model 'logistic_regresion'. Did you mean 'logistic_regression'",
    ):
        nvz.trace("logistic_regresion")


def test_unknown_dataset_and_op():
    with pytest.raises(UnknownNameError, match="Unknown dataset 'blob'. Did you mean 'blobs'"):
        nvz.datasets.load("blob")
    with pytest.raises(UnknownNameError, match="Unknown op 'sigmod'. Did you mean 'sigmoid'"):
        nvz.core.registry.get("sigmod")
    with pytest.raises(
        UnknownNameError, match="Available models: 'decision_tree', 'logistic_regression', 'mlp'"
    ):
        nvz.models.load("zzz")


def test_shape_mismatch_names_op_shapes_and_fix():
    with pytest.raises(ShapeError) as e:
        ops.linear(np.ones((2, 3)), np.ones((4, 1)), np.zeros(1))
    msg = str(e.value)
    assert msg.startswith(
        "linear: expected w with shape (3,) or (3, m) to match x (2, 3), received w (4, 1). Fix:"
    )
    with pytest.raises(ShapeError, match=r"matmul: expected a.shape\[1\] == b.shape\[0\]"):
        ops.matmul(np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(
        ShapeError,
        match=r"add: expected shapes that broadcast together, received a \(2, 3\) and b \(2,\)",
    ):
        ops.add(np.ones((2, 3)), np.ones(2))
    with pytest.raises(
        ShapeError,
        match=r"bce: expected p and y to have the same shape, received p \(4,\) and y \(4, 1\)\. "
        r"Fix: reshape y to \(4,\)",
    ):
        ops.bce(np.full(4, 0.5), np.zeros((4, 1)))
    with pytest.raises(ShapeError, match="cross_entropy: expected integer class indices"):
        ops.cross_entropy(np.zeros((2, 3)), np.array([0, 5]))


def test_unsupported_model_options_list_valid_ones():
    with pytest.raises(
        NvzError, match="logistic_regression: features=3 is not supported. Valid values: 1, 2."
    ):
        nvz.trace("logistic_regression", features=3)
    with pytest.raises(
        NvzError, match="samples='all' is not valid. Valid values: 'single', 'batch'."
    ):
        nvz.trace("logistic_regression", samples="all")
    with pytest.raises(
        NvzError, match="kind='fit' is not valid. Valid values: 'predict', 'train'."
    ):
        nvz.trace("logistic_regression", kind="fit")
    with pytest.raises(NvzError, match="iterations must be 1, 2 or 3"):
        nvz.trace("logistic_regression", kind="train", iterations=10)


def test_train_without_labels_or_loss_explains():
    def model(x, p):
        return ops.sigmoid(ops.linear(x, p["w"], p["b"]))

    p = {"w": np.ones(2), "b": np.zeros(1)}
    x = np.ones((1, 2))
    with pytest.raises(NvzError, match="kind='train' needs labels. Fix: pass y"):
        nvz.trace(model, x, params=p, kind="train", loss="bce")
    with pytest.raises(NvzError, match="kind='train' needs a loss. Fix: pass loss='bce'"):
        nvz.trace(model, x, np.ones(1), params=p, kind="train")
    with pytest.raises(NvzError, match="Unknown loss 'mse'"):
        nvz.trace(model, x, np.ones(1), params=p, kind="train", loss="mse")


def test_user_model_needs_x_and_traced_output():
    with pytest.raises(NvzError, match="x is required for a user model"):
        nvz.trace(lambda x, p: x)
    with pytest.raises(NvzError, match="no nvz.ops call produced"):
        nvz.trace(lambda x, p: x * 2, np.ones((1, 2)))
    with pytest.raises(NvzError, match="samples='batch' but x has 1 row"):
        nvz.trace(lambda x, p: ops.relu(x), np.ones((1, 2)), samples="batch")
    with pytest.raises(
        NvzError, match="logistic_regression: expected x with shape \\(n_samples, 2\\)"
    ):
        nvz.trace("logistic_regression", np.ones((1, 3)))


def test_unsupported_mlp_and_tree_options_list_valid_ones():
    with pytest.raises(NvzError, match="mlp: features=1 is not supported. Valid values: 2."):
        nvz.trace("mlp", features=1)
    with pytest.raises(NvzError, match=r"mlp: expected x with shape \(n_samples, 2\)"):
        nvz.trace("mlp", np.ones((1, 3)))
    with pytest.raises(
        NvzError, match="decision_tree: criterion='mse' is not supported. Valid values: 'gini'"
    ):
        nvz.trace("decision_tree", criterion="mse")
    with pytest.raises(
        NvzError, match="decision_tree: dataset='iris' is not supported. Valid values: 'blobs'"
    ):
        nvz.trace("decision_tree", dataset="iris")
    with pytest.raises(NvzError, match="decision_tree: features=3 is not supported"):
        nvz.trace("decision_tree", features=3)
    with pytest.raises(NvzError, match="dataset='moons' has 2 features, received features=1"):
        nvz.trace("decision_tree", dataset="moons", features=1)
    with pytest.raises(NvzError, match="learns from the whole dataset, so samples='single'"):
        nvz.trace("decision_tree", kind="train", samples="single")
    with pytest.raises(NvzError, match="kind='train' needs labels"):
        nvz.trace("decision_tree", np.ones((3, 2)), kind="train")
    with pytest.raises(NvzError, match="moons: features must be 2"):
        nvz.datasets.load("moons", features=1)
