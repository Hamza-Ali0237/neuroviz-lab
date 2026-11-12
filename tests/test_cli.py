import json

from conftest import validate
from nvzlab.cli import main


def test_list(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "models:   logistic_regression" in out and "datasets: blobs" in out and "sigmoid" in out


def test_demo_json(tmp_path, capsys):
    out = tmp_path / "out.json"
    assert (
        main(
            [
                "demo",
                "logistic_regression",
                "--kind",
                "train",
                "--samples",
                "batch",
                "--json",
                str(out),
            ]
        )
        == 0
    )
    assert "Update w and b" in capsys.readouterr().out
    validate(json.loads(out.read_text()))


def test_demo_unknown_model(capsys):
    assert main(["demo", "logistic"]) == 1
    assert "Did you mean 'logistic_regression'" in capsys.readouterr().err
