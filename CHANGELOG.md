# Changelog

## Unreleased

### Added

- Core engine: `Tape` recording with implicit autograd, `nvz.step`, `nvz.event`, SGD update events.
- Ops: `matmul`, `add`, `mul`, `linear`, `sigmoid`, `relu`, `softmax`, `tanh`, `bce`, `cross_entropy`, `mean`, `sum`, each with a forward and a backward explanation.
- `@nvz.op` decorator and `nvz.Op` class form for user-defined ops (with optional `backward_formula`/`backward_text`), with one shared registry.
- Backward steps mirror forward steps in reverse order (`nvz.step(title, backward_title=...)`); `grad_accumulate` nodes make gradient fan-in visible.
- Trace nodes carry the op's `category` and an `origin` (`builtin` or `user`).
- Trace schema v0 (`nvzlab.trace/0`), deterministic JSON serializer, `Trace.save`/`Trace.load`.
- `blobs` synthetic dataset (1 or 2 features).
- Built-in `logistic_regression` model with pretrained and initial weights.
- CLI: `nvzlab list`, `nvzlab demo <model> --json out.json`.
