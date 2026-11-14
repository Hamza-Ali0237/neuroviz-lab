import json

import numpy as np
import pytest

import nvzlab as nvz
from conftest import validate
from nvzlab.tracing.serializer import decode_array, encode_array

CASES = [
    dict(model="logistic_regression", kind=k, samples=s, features=f)
    for k in ("predict", "train")
    for s in ("single", "batch")
    for f in (1, 2)
]
CASES += [
    dict(model="mlp", kind=k, samples=s) for k in ("predict", "train") for s in ("single", "batch")
]
CASES += [
    dict(model=m, kind=k, samples=s)
    for m in ("cnn", "transformer")
    for k in ("predict", "train")
    for s in ("single", "batch")
]
CASES += [
    dict(model="decision_tree", kind="predict", samples=s, dataset=d, features=f, criterion=c)
    for s in ("single", "batch")
    for d, f in (("blobs", 1), ("blobs", 2), ("moons", 2))
    for c in ("gini", "entropy")
]
CASES += [
    dict(model="decision_tree", kind="train", dataset=d, features=f, criterion=c, iterations=i)
    for d, f in (("blobs", 1), ("blobs", 2), ("moons", 2))
    for c in ("gini", "entropy")
    for i in (1, 2, 3)
]


def _case_id(c):
    return "-".join(str(v) for v in c.values())


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_schema_valid_for_builtin(case):
    case = dict(case)
    model = case.pop("model")
    tr = nvz.trace(model, **case)
    validate(tr)
    assert tr.kind == case["kind"]
    assert tr.model["name"] == nvz.models.load(model).display_name
    assert tr.run_id.startswith(f"{model}-{case['kind']}-")
    if "samples" in case:
        assert tr.mode["samples"] == case["samples"]
    else:  # a self-trained model learns from the whole dataset
        assert tr.mode == {"samples": "batch", "n_samples": 120, "n_features": case["features"]}
    n_features = {"cnn": 192, "transformer": 6}.get(model, case.get("features", 2))
    assert tr.mode["n_features"] == n_features
    assert tr.model["provenance"] == (
        "builtin-pretrained" if case["kind"] == "predict" else "builtin-initial"
    )
    default_ds = {"mlp": "moons", "cnn": "shapes", "transformer": "tokens"}.get(model, "blobs")
    assert tr.dataset["name"] == case.get("dataset", default_ds)
    if model != "cnn":  # image datasets name their channels, not every pixel
        assert len(tr.dataset["features"]) == tr.mode["n_features"]
    assert tr.steps and all(s.layer is None or isinstance(s.layer, str) for s in tr.steps)


@pytest.mark.parametrize(
    "call",
    [
        dict(model="logistic_regression", kind="train", samples="batch", iterations=3),
        dict(model="mlp", kind="train", samples="batch", iterations=3),
        dict(model="cnn", kind="train", samples="batch", iterations=2),
        dict(model="transformer", kind="train", samples="batch", iterations=2),
        dict(model="decision_tree", kind="train", dataset="moons", iterations=3),
        dict(model="decision_tree", kind="predict", samples="batch", criterion="entropy"),
    ],
    ids=_case_id,
)
def test_determinism(call):
    call = dict(call)
    model = call.pop("model")
    a = nvz.trace(model, **call).to_json()
    b = nvz.trace(model, **call).to_json()
    assert a == b
    assert json.dumps(json.loads(a), sort_keys=True, separators=(",", ":")) == a  # keys sorted


def test_save_load_roundtrip(tmp_path):
    tr = nvz.trace("logistic_regression", kind="train")
    path = tr.save(tmp_path / "run.json")
    back = nvz.Trace.load(path)
    assert back.to_json() == tr.to_json()
    for tid, t in tr.tensors.items():
        np.testing.assert_allclose(back.tensors[tid].array, t.array.astype(np.float32))
    assert back.summary() == tr.summary()


def test_save_rejects_unknown_extension(tmp_path):
    tr = nvz.trace("logistic_regression")
    with pytest.raises(ValueError, match="unsupported extension '.txt'"):
        tr.save(tmp_path / "run.txt")


def test_load_rejects_other_schema(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"schema": "other/1"}')
    with pytest.raises(ValueError, match="expected schema"):
        nvz.Trace.load(p)


def test_buffers_are_float32_little_endian():
    arr = np.array([[1.5, -2.0]])
    encoded = encode_array(arr)
    assert np.frombuffer(__import__("base64").b64decode(encoded), dtype="<f4").tolist() == [
        1.5,
        -2.0,
    ]
    np.testing.assert_array_equal(decode_array(encoded, [1, 2]), arr)


def test_stats_rounded_and_tensor_roles():
    tr = nvz.trace("logistic_regression", kind="train", samples="batch")
    roles = {t.role for t in tr.tensors.values()}
    assert roles == {"input", "label", "param", "activation", "output", "loss", "grad", "context"}
    for t in tr.tensors.values():
        for v in t.stats.values():
            assert v == float(f"{v:.6g}")
        assert t.truncated is False and t.array is not None


def test_step_layer_round_trips_and_shows_in_summary(tmp_path):
    tr = nvz.trace("mlp", kind="train")
    layers = {s.title: s.layer for s in tr.steps}
    assert layers["Weighted sum"] == "Hidden layer" and layers["Input"] is None
    d = tr.to_dict()
    assert all("layer" in s for s in d["steps"])
    assert nvz.Trace.load(tr.save(tmp_path / "mlp.json")).steps[1].layer == "Hidden layer"
    lines = tr.summary().splitlines()
    assert lines[1].split() == ["#", "phase", "layer", "step", "ops"]
    assert "Hidden layer  Weighted sum" in lines[3]
    assert "layer" not in nvz.trace("logistic_regression").summary().splitlines()[1]


def test_summary_and_repr():
    tr = nvz.trace("logistic_regression", kind="train")
    s = tr.summary()
    assert repr(tr) == s
    for title in (
        "Input",
        "Weighted sum z = Xw + b",
        "Sigmoid",
        "Prediction",
        "Loss",
        "Gradient of the loss",
        "Gradients reach w and b",
        "Update w and b",
    ):
        assert title in s
    assert "linear (1, 2)" in s


def test_truncation_of_large_tensors():
    @nvz.op
    def test_big(x):
        return x * 2

    with nvz.Tape() as t:
        test_big(np.zeros((300, 300)))
    tr = t.to_trace()
    assert all(x.truncated and x.array is None for x in tr.tensors.values())
    assert tr.to_dict()["tensors"]["t0"]["data"] is None
    validate(tr)


def test_context_tensors_validate_and_carry_note():
    tr = nvz.trace("logistic_regression", kind="train", samples="batch", iterations=2)
    validate(tr)
    ctx = {t.name: t for t in tr.tensors.values() if t.role == "context"}
    assert set(ctx) == {
        "dataset_X",
        "dataset_y",
        "loss_landscape",
        "loss_landscape_w1",
        "loss_landscape_w2",
        "loss_path",
    }
    assert ctx["dataset_X"].shape == (120, 2) and "orientation" in ctx["dataset_X"].note
    assert ctx["loss_landscape"].shape == (41, 41)
    assert "b fixed at its initial value" in ctx["loss_landscape"].note
    # Context tensors are never used by nodes, and every other tensor has no note.
    used = {t for n in tr.nodes for t in n.inputs + n.outputs}
    assert not used & {t.id for t in ctx.values()}
    assert all(t.note is None for t in tr.tensors.values() if t.role != "context")
    assert (
        nvz.Trace.load.__name__
        and nvz.Trace.from_dict(tr.to_dict()).tensors[ctx["loss_landscape"].id].note
        == ctx["loss_landscape"].note
    )
    # predict traces record the dataset but no loss surface
    names = {
        t.name for t in nvz.trace("logistic_regression").tensors.values() if t.role == "context"
    }
    assert names == {"dataset_X", "dataset_y"}


@pytest.mark.parametrize("features", [1, 2])
def test_loss_landscape_matches_direct_evaluation(features):
    from nvzlab import ops

    tr = nvz.trace("logistic_regression", kind="train", samples="batch", features=features)
    ctx = {t.name: t.array for t in tr.tensors.values() if t.role == "context"}
    x = next(t.array for t in tr.tensors.values() if t.name == "x")
    y = next(t.array for t in tr.tensors.values() if t.name == "y")
    model = nvz.models.load("logistic_regression", features=features)
    b0 = model.initial_params["b"][0]
    a0, a1 = (
        (ctx["loss_landscape_w1"], ctx["loss_landscape_w2"])
        if features == 2
        else (ctx["loss_landscape_w"], ctx["loss_landscape_b"])
    )
    for i, j in [(0, 0), (20, 20), (40, 7)]:
        w = np.array([a0[i], a1[j]]) if features == 2 else np.array([a0[i]])
        b = np.array([b0]) if features == 2 else np.array([a1[j]])
        direct = ops.bce(ops.sigmoid(ops.linear(x, w, b)), y)
        assert ctx["loss_landscape"][i, j] == pytest.approx(float(direct), rel=1e-9)


@pytest.mark.parametrize("samples", ["single", "batch"])
def test_loss_path_is_exact_and_window_follows_the_path(samples):
    from nvzlab import ops

    tr = nvz.trace("logistic_regression", kind="train", samples=samples, iterations=3)
    assert tr.meta["lr"] == nvz.models.load("logistic_regression").default_lr == 0.5
    ctx = {t.name: t.array for t in tr.tensors.values() if t.role == "context"}
    x = next(t.array for t in tr.tensors.values() if t.name == "x")
    y = next(t.array for t in tr.tensors.values() if t.name == "y")
    # The visited parameter sets: initial, then the `after` of every update.
    first: dict = {}
    for t in tr.tensors.values():
        if t.role == "param":
            first.setdefault(t.name, t.array)
    path = [(first["w"], first["b"])]
    for e in tr.events:
        if e.kind == "sgd_update" and e.payload["param"] == "w":
            b_ev = next(
                ev
                for ev in tr.events
                if ev.kind == "sgd_update" and ev.step == e.step and ev.payload["param"] == "b"
            )
            path.append(
                (tr.tensors[e.payload["after"]].array, tr.tensors[b_ev.payload["after"]].array)
            )
    assert len(path) == 4 and ctx["loss_path"].shape == (4,)
    direct = [float(ops.bce(ops.sigmoid(ops.linear(x, w, b)), y)) for w, b in path]
    np.testing.assert_allclose(ctx["loss_path"], direct, rtol=1e-6)
    recorded = [float(t.array) for t in tr.tensors.values() if t.role == "loss"]
    np.testing.assert_allclose(ctx["loss_path"][:3], recorded, rtol=1e-6)
    assert all(a > b for a, b in zip(direct, direct[1:], strict=False)), direct  # downhill
    # Window: centred on the path, half-width max(3 × extent, 0.25) per axis.
    coords = np.array([[w[0], w[1]] for w, _ in path])
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    half = np.maximum(3 * (hi - lo), 0.25)
    for k, name in enumerate(("w1", "w2")):
        g = ctx[f"loss_landscape_{name}"]
        assert g[0] == pytest.approx((lo[k] + hi[k]) / 2 - half[k])
        assert g[-1] == pytest.approx((lo[k] + hi[k]) / 2 + half[k])
    note = next(t.note for t in tr.tensors.values() if t.name == "loss_landscape")
    assert "Window centred on the training path" in note and "w1 in [" in note


def test_user_lr_wins_over_default_and_functions_default_to_0_1():
    tr = nvz.trace("logistic_regression", kind="train", lr=0.05)
    assert tr.meta["lr"] == 0.05
    import numpy as np

    from nvzlab import ops

    def model(x, p):
        return ops.sigmoid(ops.linear(x, p["w"], p["b"]))

    tr = nvz.trace(
        model,
        np.array([[1.0, 2.0]]),
        np.array([1.0]),
        params={"w": np.zeros(2), "b": np.zeros(1)},
        kind="train",
        loss="bce",
    )
    assert tr.meta["lr"] == 0.1
    ctx = {t.name: t.array for t in tr.tensors.values() if t.role == "context"}
    assert ctx["loss_path"].shape == (2,) and ctx["loss_path"][1] < ctx["loss_path"][0]
