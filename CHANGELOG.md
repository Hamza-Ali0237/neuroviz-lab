# Changelog

## Unreleased

### Added

- Core engine: `Tape` recording with implicit autograd, `nvz.step`, `nvz.event`, SGD update events.
- Ops: `matmul`, `add`, `linear`, `sigmoid`, `relu`, `softmax`, `bce`, `cross_entropy`, `mean`, `sum`.
- `@nvz.op` decorator and `nvz.Op` class form for user-defined ops, with one shared registry.
- Trace schema v0 (`nvzlab.trace/0`), deterministic JSON serializer, `Trace.save`/`Trace.load`.
- `blobs` synthetic dataset (1 or 2 features).
- Built-in `logistic_regression` model with pretrained and initial weights.
- CLI: `nvzlab list`, `nvzlab demo <model> --json out.json`.
