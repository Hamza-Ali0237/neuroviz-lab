import type { TraceView, Tensor } from "../trace/load";

/** One layer of the network as seen at a step, for the selected sample. */
export interface LayerView {
  label: string;
  size: number;
  /** Values for the selected sample, or null when this layer has not been computed yet at this step. */
  values: number[] | null;
  /** Tensor holding the values (row = sample), for selection. */
  tensorId: string | null;
  /** Largest |value| of the whole tensor, for the colour scale. */
  absMax: number;
}

/** The connection between two layers: a weight matrix (in × out) or, on backward steps, its gradient. */
export interface EdgeView {
  tensor: Tensor;
  /** True when `tensor` is ∂L/∂W rather than W. */
  grad: boolean;
  /** Name of the weight (W1, W2). */
  name: string;
}

export interface MLPView {
  layers: LayerView[];
  edges: EdgeView[];
  /** Number of parameter updates applied at or before this step (= index of the current prob_grid). */
  reached: number;
  backward: boolean;
}

const absMaxOf = (t: Tensor) => Math.max(Math.abs(t.stats.min), Math.abs(t.stats.max), 1e-12);

/**
 * Read the 2 → 4 → 2 MLP out of the trace at a step: which layers have been computed, their
 * values for the sample, and the weights (or gradients) on the connections. Works for any
 * `linear → relu → linear → softmax` forward written with nvz.ops; returns null otherwise.
 */
export function mlpView(trace: TraceView, stepIndex: number, sampleIndex: number): MLPView | null {
  const step = trace.steps[stepIndex];
  const it = step.iteration;
  const forward = trace.raw.nodes.filter((n) => n.phase === "forward" && trace.steps[n.step].iteration === it);
  const ops = forward.map((n) => n.op);
  const i1 = ops.indexOf("linear");
  const iR = ops.indexOf("relu", i1);
  const i2 = ops.indexOf("linear", iR);
  const iS = ops.indexOf("softmax", i2);
  if (i1 < 0 || iR < 0 || i2 < 0 || iS < 0) return null;
  const [lin1, relu, lin2, soft] = [i1, iR, i2, iS].map((i) => forward[i]);
  const x = trace.byRole("input", "x");
  if (!x?.values || x.shape.length !== 2) return null;
  const row = (t: Tensor, i: number) => {
    const cols = t.shape[t.shape.length - 1];
    return Array.from({ length: cols }, (_, j) => t.values![i * cols + j]);
  };
  const done = (n: { step: number }) => n.step <= stepIndex;
  const out = (n: { outputs: string[] }) => trace.tensor(n.outputs[0]);
  const hidden = done(relu) ? out(relu) : done(lin1) ? out(lin1) : null;
  const output = done(soft) ? out(soft) : done(lin2) ? out(lin2) : null;
  const layer = (label: string, size: number, t: Tensor | null): LayerView => ({
    label,
    size,
    values: t?.values ? row(t, sampleIndex) : null,
    tensorId: t?.id ?? null,
    absMax: t ? absMaxOf(t) : 1,
  });
  const W1 = trace.tensor(lin1.inputs[1]);
  const W2 = trace.tensor(lin2.inputs[1]);
  const layers = [
    layer("input x", x.shape[1], x),
    layer(done(relu) ? "hidden h" : "hidden z₁", W1.shape[1], hidden),
    layer(done(soft) ? "output p" : "logits z₂", W2.shape[1], output),
  ];
  // Edges: the weights used by this iteration's forward pass. On backward steps, the gradient
  // of each weight once its backward node has run (W2 first, then W1).
  const backward = step.phase === "backward";
  const gradOf = (wid: string): Tensor | null => {
    for (const n of trace.raw.nodes) {
      if (n.phase !== "backward" || trace.steps[n.step].iteration !== it || n.step > stepIndex) continue;
      const k = (n.attrs.grad_of ?? []).indexOf(wid);
      if (k >= 0) return trace.tensor(n.outputs[k]);
    }
    return null;
  };
  // At an update step the edges show the updated weights.
  const params = trace.paramsAt(stepIndex);
  const nameOf = (t: Tensor) => t.name ?? "W";
  const edge = (W: Tensor): EdgeView => {
    const g = backward ? gradOf(W.id) : null;
    if (g) return { tensor: g, grad: true, name: nameOf(W) };
    const current = step.phase === "update" ? params.get(nameOf(W)) : null;
    return { tensor: current ?? W, grad: false, name: nameOf(W) };
  };
  const reached = trace.raw.events.filter((e) => e.kind === "sgd_update" && e.step <= stepIndex && e.payload.param === "W1").length;
  return { layers, edges: [edge(W1), edge(W2)], reached, backward };
}

/** The class-1 probability grid current at a step (`prob_grid_<reached>`), with its axes. */
export function probGridAt(trace: TraceView, reached: number): { grid: Tensor; ax1: Float32Array; ax2: Float32Array } | null {
  const grid = trace.context(`prob_grid_${reached}`) ?? trace.context("prob_grid_0");
  const ax1 = trace.context("prob_grid_x1")?.values;
  const ax2 = trace.context("prob_grid_x2")?.values;
  if (!grid?.values || !ax1 || !ax2) return null;
  return { grid, ax1, ax2 };
}
