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
    assert tr.kind == case["kind"] and tr.model["name"] == model
    if "samples" in case:
        assert tr.mode["samples"] == case["samples"]
    else:  # a self-trained model learns from the whole dataset
        assert tr.mode == {"samples": "batch", "n_samples": 120, "n_features": case["features"]}
    assert tr.mode["n_features"] == case.get("features", 2)
    assert tr.model["provenance"] == (
        "builtin-pretrained" if case["kind"] == "predict" else "builtin-initial"
    )
    assert tr.dataset["name"] == case.get("dataset", "moons" if model == "mlp" else "blobs")
    assert len(tr.dataset["features"]) == tr.mode["n_features"]
    assert tr.steps and all(s.layer is None or isinstance(s.layer, str) for s in tr.steps)


@pytest.mark.parametrize(
    "call",
    [
        dict(model="logistic_regression", kind="train", samples="batch", iterations=3),
        dict(model="mlp", kind="train", samples="batch", iterations=3),
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
    assert roles == {"input", "label", "param", "activation", "output", "loss", "grad"}
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
