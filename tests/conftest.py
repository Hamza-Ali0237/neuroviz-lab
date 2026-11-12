import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

SCHEMA = json.loads((Path(__file__).parents[1] / "src/nvzlab/tracing/schema_v0.json").read_text())
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)


def validate(trace) -> None:
    """Validate a Trace (or its dict) against schema v0; raises on error."""
    d = trace if isinstance(trace, dict) else trace.to_dict()
    errors = sorted(VALIDATOR.iter_errors(d), key=lambda e: list(e.path))
    assert not errors, "\n".join(f"{list(e.path)}: {e.message}" for e in errors)


def gradcheck(op, inputs, diff=None, eps=1e-6, **attrs):
    """Central-difference gradient check for every differentiable input of ``op``."""
    rng = np.random.default_rng(0)
    inputs = [np.asarray(a, dtype=np.float64) for a in inputs]
    out, ctx = op.forward(*inputs, **attrs)
    g_out = rng.normal(size=out.shape)
    analytic = op.backward(ctx, g_out)
    assert len(analytic) == len(inputs)
    diff = range(len(inputs)) if diff is None else diff

    def scalar_loss(args):
        return float(np.sum(g_out * op.forward(*args, **attrs)[0]))

    for i in diff:
        assert analytic[i] is not None, f"input {i} of {op.name} got no gradient"
        num = np.zeros_like(inputs[i])
        for idx in np.ndindex(*inputs[i].shape):
            plus = [a.copy() for a in inputs]
            minus = [a.copy() for a in inputs]
            plus[i][idx] += eps
            minus[i][idx] -= eps
            num[idx] = (scalar_loss(plus) - scalar_loss(minus)) / (2 * eps)
        np.testing.assert_allclose(
            analytic[i], num, rtol=1e-5, atol=1e-7, err_msg=f"{op.name} input {i}"
        )


@pytest.fixture
def rng():
    return np.random.default_rng(0)
