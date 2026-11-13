"""Viewer: self-contained HTML export and display of a trace.

The built viewer (one JS file and one CSS file in ``static/``) is inlined together with the
trace JSON, so the resulting page works offline and needs no server.

Examples
--------
>>> import nvzlab as nvz
>>> tr = nvz.trace("logistic_regression")
>>> html = nvz.viewer.to_html(tr)
>>> html.startswith("<!doctype html>")
True
"""

from __future__ import annotations

import sys
import tempfile
import webbrowser
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nvzlab.tracing.records import Trace

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nvzlab · {title}</title>
<style>{css}</style>
</head>
<body>
<div id="root" data-dim="{dim}"></div>
<script id="nvz-trace" type="application/json">{trace}</script>
<script type="module">{js}</script>
</body>
</html>
"""


def _static(name: str) -> str:
    path = resources.files("nvzlab.viewer") / "static" / name
    if not path.is_file():
        raise RuntimeError(
            f"Viewer bundle {name!r} is missing from the installed package. "
            "Fix: reinstall neuroviz-lab, or run `npm run build` in frontend/ for a checkout."
        )
    return path.read_text(encoding="utf-8")


def to_html(trace: Trace, dim: str = "2d") -> str:
    """The trace as one self-contained HTML page (JS, CSS and JSON inlined).

    ``dim`` picks the renderer the page opens with (``"2d"`` or ``"3d"``); the viewer's
    toggle switches between them at any time.

    Examples
    --------
    >>> html = nvz.viewer.to_html(nvz.trace("logistic_regression"))  # doctest: +SKIP
    """
    # "<" never appears unescaped inside the JSON, so "</script>" cannot end the tag early.
    payload = trace.to_json().replace("<", "\\u003c")
    if dim not in ("2d", "3d"):
        raise ValueError(f"dim must be '2d' or '3d', received {dim!r}.")
    return _PAGE.format(
        title=trace.run_id,
        css=_static("nvzlab.css"),
        trace=payload,
        js=_static("nvzlab.js"),
        dim=dim,
    )


def in_notebook() -> bool:
    """True when running inside a Jupyter kernel with a display frontend."""
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def show(trace: Trace, height: int = 720, dim: str = "2d") -> Path | None:
    """Display ``trace``: inline in Jupyter, otherwise in the default browser.

    Outside a notebook the page is written to a temporary file whose path is printed and
    returned, so it can be reopened or shared.

    Examples
    --------
    >>> nvz.viewer.show(nvz.trace("logistic_regression"))  # doctest: +SKIP
    """
    html = to_html(trace, dim)
    if in_notebook():
        from IPython.display import HTML, display  # type: ignore[import-not-found]

        srcdoc = html.replace("&", "&amp;").replace('"', "&quot;")
        display(
            HTML(
                f'<iframe srcdoc="{srcdoc}" style="width:100%;height:{height}px;border:0" '
                'sandbox="allow-scripts"></iframe>'
            )
        )
        return None
    path = Path(tempfile.mkdtemp(prefix="nvzlab-")) / f"{trace.run_id}.html"
    path.write_text(html, encoding="utf-8")
    print(f"nvzlab: viewer written to {path}", file=sys.stderr)
    webbrowser.open(path.as_uri())
    return path


__all__ = ["in_notebook", "show", "to_html"]
