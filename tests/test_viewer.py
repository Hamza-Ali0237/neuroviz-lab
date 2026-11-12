import re

import nvzlab as nvz
from nvzlab import viewer
from nvzlab.cli import main

EXTERNAL = re.compile(r"""(src|href)=["']?\s*(https?:)?//|url\(\s*["']?https?:|@import""")


def test_save_html_is_self_contained(tmp_path):
    tr = nvz.trace("logistic_regression", kind="train", samples="batch")
    out = tr.save(tmp_path / "run.html")
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert 'id="nvz-trace"' in html and tr.run_id in html
    assert '"schema":"nvzlab.trace/0"' in html
    assert "</script>" not in tr.to_json().replace("<", "\\u003c")
    assert "<\\/script" not in html.split('id="nvz-trace"')[1].split("</script>")[0]
    assert not EXTERNAL.search(html), "page references an external resource"
    assert "nvz-app" in html  # the built bundle is inlined


def test_html_embeds_json_safely():
    tr = nvz.trace("logistic_regression")
    html = viewer.to_html(tr)
    payload = html.split('type="application/json">')[1].split("</script>")[0]
    assert "<" not in payload
    import json

    assert json.loads(payload)["run_id"] == tr.run_id


def test_show_outside_notebook_writes_file(monkeypatch, capsys):
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
    path = nvz.trace("logistic_regression").show()
    assert path is not None and path.exists() and path.suffix == ".html"
    assert opened == [path.as_uri()]
    assert str(path) in capsys.readouterr().err


def test_nvz_show_returns_trace(monkeypatch):
    monkeypatch.setattr("webbrowser.open", lambda url: True)
    tr = nvz.show("logistic_regression", samples="batch")
    assert tr.kind == "predict" and tr.mode["n_samples"] == 4


def test_cli_open_and_demo_html(tmp_path, monkeypatch):
    json_path = tmp_path / "t.json"
    assert main(["demo", "logistic_regression", "--json", str(json_path)]) == 0
    html_path = tmp_path / "t.html"
    assert main(["open", str(json_path), "--html", str(html_path)]) == 0
    assert "nvz-trace" in html_path.read_text()
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
    assert main(["open", str(json_path)]) == 0
    assert len(opened) == 1
