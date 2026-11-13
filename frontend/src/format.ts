/** Number formatting shared by every renderer. */

/** Short form for grids and worked arithmetic: 4 significant digits, no trailing zeros. */
export function fmt(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "·";
  if (!Number.isFinite(v)) return v > 0 ? "∞" : "−∞";
  if (v === 0) return "0";
  const s = Math.abs(v) >= 1e4 || Math.abs(v) < 1e-3 ? v.toExponential(3) : v.toPrecision(4);
  return s.replace(/\.?0+(e|$)/, "$1").replace("-", "−");
}

/** Full precision for hover titles and the inspector. */
export function fmtFull(v: number | null | undefined): string {
  if (v === null || v === undefined) return "truncated";
  return String(v);
}

/** Wrap negative numbers in parentheses inside a product. */
export function fmtTerm(v: number | null | undefined): string {
  const s = fmt(v);
  return s.startsWith("−") ? `(${s})` : s;
}

export function shapeStr(shape: number[]): string {
  return shape.length ? `(${shape.join(", ")})` : "()";
}

export function sub(i: number): string {
  return String(i).replace(/\d/g, (d) => "₀₁₂₃₄₅₆₇₈₉"[Number(d)]);
}

export const PHASES = ["forward", "loss", "backward", "update"] as const;

/** A signed term for equations: " − 0.1" or " + 0.1" (never "+ −0.1"). */
export function signed(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return " + ·";
  return (v < 0 ? " − " : " + ") + fmt(Math.abs(v));
}

/** Evenly spaced "nice" tick values covering [lo, hi]. */
export function ticks(lo: number, hi: number, count = 6): number[] {
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

/** Diverging tint shared by the 2D grids and the 3D cells: blue positive, orange negative. */
export const POS_RGB: [number, number, number] = [37, 99, 235];
export const NEG_RGB: [number, number, number] = [234, 88, 12];
export function tintAlpha(v: number, absMax: number): number {
  return Math.min(1, Math.abs(v) / Math.max(absMax, 1e-12)) * 0.55;
}
export function tint(v: number, absMax: number): string {
  const [r, g, b] = v >= 0 ? POS_RGB : NEG_RGB;
  return `rgba(${r}, ${g}, ${b}, ${tintAlpha(v, absMax)})`;
}

/** Human labels for event kinds and payload keys shown in the UI. */
const EVENT_LABELS: Record<string, string> = {
  sgd_update: "Parameter update",
  sgd_skipped: "No update (no gradient)",
  backward_stopped: "Backward pass stops here",
  input: "Input",
  prediction: "Prediction",
  split_candidate: "Split candidate",
  split_chosen: "Split chosen",
  leaf: "Leaf",
};
const KEY_LABELS: Record<string, string> = {
  n_features: "Features",
  n_samples: "Samples",
  predicted_class: "Predicted class",
  threshold: "Threshold",
  param: "Parameter",
  lr: "Learning rate",
  before: "Before",
  grad: "Gradient",
  after: "After",
  reason: "Reason",
  node: "Node",
  op: "Op",
};
export function eventLabel(kind: string): string {
  return EVENT_LABELS[kind] ?? kind.replace(/_/g, " ");
}
export function keyLabel(key: string): string {
  return KEY_LABELS[key] ?? key.replace(/_/g, " ");
}
const PLURALS: Record<string, string> = {
  sgd_update: "parameter updates",
  sgd_skipped: "params skipped (no gradient)",
  split_candidate: "split candidates",
  leaf: "leaves",
};
/** "sgd_update, sgd_update" → "2 parameter updates"; mixed kinds are listed once each. */
export function eventSummary(kinds: string[]): string {
  const counts = new Map<string, number>();
  for (const k of kinds) counts.set(k, (counts.get(k) ?? 0) + 1);
  return [...counts.entries()]
    .map(([k, n]) => (n === 1 ? eventLabel(k) : `${n} ${PLURALS[k] ?? `× ${eventLabel(k)}`}`))
    .join(", ");
}
