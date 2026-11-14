---
name: Bug report
about: Something is wrong, crashes, or shows a number that cannot be right
title: ""
labels: bug
assignees: ""
---

## What happened

A clear description of the problem. If a value on screen looks wrong, say which step,
which tensor and which index, and what you expected instead.

## How to reproduce

The smallest call that shows it, for example:

```python
import nvzlab as nvz

tr = nvz.trace("mlp", kind="train", samples="batch")
```

If a saved trace helps, attach the `.json` written by `tr.save("run.json")`.

## What you expected

## Environment

- neuroviz-lab version (`python -c "import nvzlab; print(nvzlab.__version__)"`):
- Python version and OS:
- Where the viewer ran (Jupyter, browser from a script, `nvzlab demo`) and the browser:

## Anything else

Error message in full, screenshots, or the step list from `print(tr)`.
