import json

import numpy as np
import pytest

import nvzlab as nvz
from conftest import validate
from nvzlab.tracing.serializer import decode_array, encode_array

CASES = [
    dict(kind=k, samples=s, features=f)
    for k in ("predict", "train")
    for s in ("single", "batch")
    for f in (1, 2)
]


@pytest.mark.parametrize(
    "case", CASES, ids=lambda c: f"{c['kind']}-{c['samples']}-f{c['features']}"
)
def test_schema_valid_for_builtin(case):
    tr = nvz.trace("logistic_regression", **case)
    validate(tr)
    assert tr.mode["samples"] == case["samples"] and tr.mode["n_features"] == case["features"]
    assert tr.model["provenance"] == (
        "builtin-pretrained" if case["kind"] == "predict" else "builtin-initial"
    )
    assert tr.dataset["name"] == "blobs" and len(tr.dataset["features"]) == case["features"]


def test_determinism():
    a = nvz.trace("logistic_regression", kind="train", samples="batch", iterations=3).to_json()
    b = nvz.trace("logistic_regression", kind="train", samples="batch", iterations=3).to_json()
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


def test_save_html_and_show_not_yet(tmp_path):
    tr = nvz.trace("logistic_regression")
    with pytest.raises(NotImplementedError, match="HTML viewer arrives in the next release step"):
        tr.save(tmp_path / "run.html")
    with pytest.raises(NotImplementedError, match="HTML viewer arrives in the next release step"):
        tr.show()
    with pytest.raises(NotImplementedError, match="HTML viewer arrives in the next release step"):
        nvz.show("logistic_regression")
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
