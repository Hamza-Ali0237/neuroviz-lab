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
