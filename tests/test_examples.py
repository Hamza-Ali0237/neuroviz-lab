import json
import os
import runpy
from pathlib import Path

import pytest

from conftest import validate

EXAMPLES = sorted((Path(__file__).parents[1] / "examples").glob("*.py"))


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs_and_writes_valid_traces(script, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script), run_name="__main__")
    written = sorted(tmp_path.glob("*.json"))
    assert written, "example wrote no trace"
    for p in written:
        validate(json.loads(p.read_text()))


def test_opaque_example_marks_non_differentiable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(EXAMPLES[2]), run_name="__main__")
    d = json.loads((tmp_path / "03_opaque_op.json").read_text())
    node = next(n for n in d["nodes"] if n["op"] == "clamp_half")
    assert node["explain"] is None and node["differentiable"] is False
    assert any(e["kind"] == "backward_stopped" for e in d["events"])
    assert not any(n["phase"] == "backward" and n["forward_of"] != "n3" for n in d["nodes"][4:])


def test_readme_quickstart_runs(tmp_path, monkeypatch):
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    block = readme.split("## Quickstart")[1].split("```python")[1].split("```")[0]
    assert len([line for line in block.strip().splitlines() if line.strip()]) <= 5
    monkeypatch.chdir(tmp_path)
    exec(compile(block, "README quickstart", "exec"), {})
    assert (tmp_path / "run.json").exists()
    assert os.path.getsize(tmp_path / "run.json") > 0
