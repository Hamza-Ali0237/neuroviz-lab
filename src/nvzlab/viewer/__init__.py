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
<div id="root" data-dim="{dim}" data-capture="{capture}" data-capture-seconds="{seconds}"></div>
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


CAPTURES = (None, "square", "vertical")


def check_capture(capture: str | None, seconds: float) -> None:
    """Raise a clear error for an invalid ``capture`` or ``capture_step_seconds``."""
    if capture not in CAPTURES:
        raise ValueError(
            f"capture must be None, 'square' (1080×1080) or 'vertical' (1080×1920), "
            f"received {capture!r}."
        )
    if not isinstance(seconds, (int, float)) or not 0.2 <= seconds <= 30:
        raise ValueError(
            f"capture_step_seconds must be a number between 0.2 and 30, received {seconds!r}."
        )


def to_html(
    trace: Trace, dim: str = "2d", capture: str | None = None, capture_step_seconds: float = 2.0
) -> str:
    """The trace as one self-contained HTML page (JS, CSS and JSON inlined).

    ``dim`` picks the renderer the page opens with (``"2d"`` or ``"3d"``); the viewer's
    toggle switches between them at any time. ``capture="square"`` (1080×1080) or
    ``"vertical"`` (1080×1920) opens the page in capture mode: no controls, large type,
    auto-play through every step (``capture_step_seconds`` each) and a clean loop, for
    screen recording. The same flags work in the URL: ``?capture=square&dim=3d``.

    Examples
    --------
    >>> html = nvz.viewer.to_html(nvz.trace("logistic_regression"))  # doctest: +SKIP
    >>> html = nvz.viewer.to_html(nvz.trace("mlp"), capture="square")  # doctest: +SKIP
    """
    # "<" never appears unescaped inside the JSON, so "</script>" cannot end the tag early.
    payload = trace.to_json().replace("<", "\\u003c")
    if dim not in ("2d", "3d"):
        raise ValueError(f"dim must be '2d' or '3d', received {dim!r}.")
    check_capture(capture, capture_step_seconds)
    return _PAGE.format(
        title=trace.run_id,
        css=_static("nvzlab.css"),
        trace=payload,
        js=_static("nvzlab.js"),
        dim=dim,
        capture=capture or "",
        seconds=capture_step_seconds,
    )


def in_notebook() -> bool:
    """True when running inside a Jupyter kernel with a display frontend."""
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def show(
    trace: Trace,
    height: int = 720,
    dim: str = "2d",
    capture: str | None = None,
    capture_step_seconds: float = 2.0,
) -> Path | None:
    """Display ``trace``: inline in Jupyter, otherwise in the default browser.

    Outside a notebook the page is written to a temporary file whose path is printed and
    returned, so it can be reopened or shared. ``capture`` opens capture mode (see
    :func:`to_html`).

    Examples
    --------
    >>> nvz.viewer.show(nvz.trace("logistic_regression"))  # doctest: +SKIP
    >>> nvz.viewer.show(nvz.trace("mlp"), capture="vertical")  # doctest: +SKIP
    """
    html = to_html(trace, dim, capture, capture_step_seconds)
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


__all__ = ["CAPTURES", "check_capture", "in_notebook", "show", "to_html"]
