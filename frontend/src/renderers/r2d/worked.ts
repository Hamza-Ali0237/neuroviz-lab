/**
 * Worked arithmetic for one output cell of a node, built only from trace values.
 * Returns null when no rule exists for the op (the UI then shows values only).
 */
import type { TraceView } from "../../trace/load";
import type { NodeRecord } from "../../trace/types";
import { fmt, fmtTerm, signed, sub } from "../../format";

export interface WorkedLine {
  /** Left-hand side, e.g. "z₀" or "∂L/∂w₁". */
  lhs: string;
  /** The expression with numbers substituted. */
  expr: string;
  /** The value of the output cell as stored in the trace (or, for intermediate lines, computed here from trace values). */
  value: number;
  /** True for intermediate lines whose value is computed in the viewer, not stored in the trace. */
  derived?: boolean;
}

const MAX_TERMS = 12;

/** Symbols for unnamed tensors, by the op that produced them. */
const SYMBOL: Record<string, string> = { linear: "z", matmul: "z", sigmoid: "σ", tanh: "a", relu: "a", add: "c", mul: "c", conv2d: "z", maxpool2d: "p", flatten: "f", softmax: "s", layernorm: "x̂", embedding: "e", scale: "s", transpose: "kᵀ", mean: "m" };
/** Q, K, V for the projections named after their weight. */
const BY_WEIGHT: Record<string, string> = { W_q: "Q", W_k: "K", W_v: "V" };

function symbol(trace: TraceView, tid: string): string {
  const t = trace.tensor(tid);
  if (t.name) return t.name;
  const p = trace.producer(tid);
  if (p?.op === "matmul") {
    const w = trace.tensor(p.inputs[1]).name;
    if (w && BY_WEIGHT[w]) return BY_WEIGHT[w];
  }
  return (p && SYMBOL[p.op]) ?? trace.label(tid);
}

function val(trace: TraceView, tid: string, index: number[]): number {
  const v = trace.value(tid, index);
  if (v === null) throw new Error(`tensor ${tid} is truncated`);
  return v;
}

/** Index into a broadcast operand for an output index (NumPy rules). */
function bcastIndex(shape: number[], outIndex: number[]): number[] {
  const off = outIndex.length - shape.length;
  return shape.map((d, i) => (d === 1 ? 0 : outIndex[i + off]));
}

function sumTerms(terms: string[], joiner = " + "): string {
  if (terms.length <= MAX_TERMS) return terms.join(joiner);
  return `${terms.slice(0, MAX_TERMS).join(joiner)} + … (${terms.length} terms)`;
}

function idxName(name: string, index: number[]): string {
  return index.length ? `${name}${index.map(sub).join(",")}` : name;
}

export function worked(
  trace: TraceView,
  node: NodeRecord,
  outId: string,
  index: number[],
): WorkedLine[] | null {
  try {
    return node.phase === "backward" ? workedBackward(trace, node, outId, index) : workedForward(trace, node, outId, index);
  } catch {
    return null;
  }
}

/** Terms of a (…, i, k) · (k, j) or (…, k, j) product for output index (…, i, j). */
function matmulTerms(trace: TraceView, xId: string, wId: string, index: number[]): string[] {
  const xs = trace.tensor(xId).shape;
  const ws = trace.tensor(wId).shape;
  const lead = index.slice(0, -2);
  const [i, j] = index.slice(-2);
  const f = xs[xs.length - 1];
  const terms: string[] = [];
  for (let k = 0; k < f; k++) {
    const xv = val(trace, xId, [...lead.slice(lead.length - (xs.length - 2)), i, k]);
    const wv = val(trace, wId, ws.length === 1 ? [k] : [...lead.slice(lead.length - (ws.length - 2)), k, j]);
    terms.push(`${fmtTerm(xv)}·${fmtTerm(wv)}`);
  }
  return terms;
}

function workedForward(trace: TraceView, node: NodeRecord, outId: string, index: number[]): WorkedLine[] | null {
  const value = val(trace, outId, index);
  const ins = node.inputs;
  const shape = (t: string) => trace.tensor(t).shape;
  const sym = symbol(trace, outId);
  switch (node.op) {
    case "linear":
    case "matmul": {
      const [xId, wId] = ins;
      const bId = node.op === "linear" ? ins[2] : null;
      const wIs1d = shape(wId).length === 1;
      const idx = wIs1d ? [...index, 0] : index; // z (n,) with a 1-D w behaves like (n, 1)
      let expr = sumTerms(matmulTerms(trace, xId, wId, idx));
      if (bId) {
        const bShape = shape(bId);
        const bv = val(trace, bId, bShape.length === 0 ? [] : [wIs1d ? 0 : index[index.length - 1]]);
        expr += signed(bv);
      }
      return [{ lhs: idxName(sym, index), expr, value }];
    }
    case "sigmoid": {
      const z = val(trace, ins[0], index);
      return [{ lhs: idxName("σ", index), expr: `1 / (1 + e^(${fmt(-z)}))`, value }];
    }
    case "relu": {
      const z = val(trace, ins[0], index);
      return [{ lhs: idxName("relu", index), expr: `max(0, ${fmt(z)})`, value }];
    }
    case "tanh": {
      const z = val(trace, ins[0], index);
      return [{ lhs: idxName("tanh", index), expr: `tanh(${fmt(z)})`, value }];
    }
    case "add":
    case "mul": {
      const a = val(trace, ins[0], bcastIndex(shape(ins[0]), index));
      const b = val(trace, ins[1], bcastIndex(shape(ins[1]), index));
      const op = node.op === "add" ? "+" : "·";
      return [{ lhs: idxName("c", index), expr: `${fmtTerm(a)} ${op} ${fmtTerm(b)}`, value }];
    }
    case "scale": {
      const x = val(trace, ins[0], index);
      return [{ lhs: idxName(sym, index), expr: `${fmt(node.attrs.c as number)} · ${fmtTerm(x)}`, value }];
    }
    case "transpose": {
      const axes = node.attrs.axes as number[];
      const src = axes.map((_, i) => index[axes.indexOf(i)]);
      return [{ lhs: idxName(sym, index), expr: `${idxName(symbol(trace, ins[0]), src)} = ${fmt(val(trace, ins[0], src))}`, value }];
    }
    case "embedding": {
      const [idsId, tableId] = ins;
      const pos = index.slice(0, -1);
      const id = val(trace, idsId, pos);
      const tab = symbol(trace, tableId);
      return [{ lhs: idxName(sym, index), expr: `${tab}[${fmt(id)}, ${index[index.length - 1]}] (token id ${fmt(id)} picks row ${fmt(id)})`, value }];
    }
    case "softmax": {
      // One row: every exponential, their sum, and this cell's share.
      const x = trace.tensor(ins[0]);
      const axis = (((node.attrs.axis as number | undefined) ?? -1) + x.shape.length) % x.shape.length;
      const n = x.shape[axis];
      const row = (k: number) => val(trace, ins[0], index.map((d, a) => (a === axis ? k : d)));
      const xs = Array.from({ length: n }, (_, k) => row(k));
      const es = xs.map(Math.exp);
      const total = es.reduce((a, b) => a + b, 0);
      const k0 = index[axis];
      const lines: WorkedLine[] = [
        { lhs: "Σ eˣ", expr: sumTerms(xs.map((v) => `e^(${fmt(v)})`)) + " = " + sumTerms(es.map((e) => fmt(e))), value: total, derived: true },
        { lhs: idxName(sym, index), expr: `e^(${fmt(xs[k0])}) / Σ = ${fmt(es[k0])} / ${fmt(total)}`, value },
      ];
      return lines;
    }
    case "layernorm": {
      // One position: mean, variance, the normalised value, then γ and β.
      const [xId, gId, bId] = ins;
      const d = shape(xId)[shape(xId).length - 1];
      const j = index[index.length - 1];
      const xs = Array.from({ length: d }, (_, k) => val(trace, xId, [...index.slice(0, -1), k]));
      const mu = xs.reduce((a, b) => a + b, 0) / d;
      const vr = xs.reduce((a, b) => a + (b - mu) ** 2, 0) / d;
      const eps = (node.attrs.eps as number | undefined) ?? 1e-5;
      const xhat = (xs[j] - mu) / Math.sqrt(vr + eps);
      const g = val(trace, gId, [j]);
      const b = val(trace, bId, [j]);
      return [
        { lhs: "μ", expr: `(${sumTerms(xs.map(fmtTerm))}) / ${d}`, value: mu, derived: true },
        { lhs: "σ²", expr: `mean((x − ${fmt(mu)})²) over ${d} values`, value: vr, derived: true },
        { lhs: idxName("x̂", index), expr: `(${fmt(xs[j])} − ${fmt(mu)}) / √(${fmt(vr)} + ${eps})`, value: xhat, derived: true },
        { lhs: idxName(sym, index), expr: `γ${sub(j)}·x̂ + β${sub(j)} = ${fmtTerm(g)}·${fmtTerm(xhat)}${signed(b)}`, value },
      ];
    }
    case "conv2d": {
      // One output cell: 9 products per input channel, their partial sums, then the bias.
      const [xId, wId, bId] = ins;
      const [n, o, i, j] = index;
      const [, C, H, W] = shape(xId);
      const [, , kh, kw] = shape(wId);
      const s = (node.attrs.stride as number | undefined) ?? 1;
      const p = (node.attrs.padding as number | undefined) ?? 0;
      const lines: WorkedLine[] = [];
      const parts: number[] = [];
      for (let c = 0; c < C; c++) {
        const terms: string[] = [];
        let part = 0;
        for (let u = 0; u < kh; u++) {
          for (let v = 0; v < kw; v++) {
            const h = i * s - p + u;
            const w = j * s - p + v;
            const xv = h >= 0 && h < H && w >= 0 && w < W ? val(trace, xId, [n, c, h, w]) : 0;
            const wv = val(trace, wId, [o, c, u, v]);
            terms.push(`${fmtTerm(xv)}·${fmtTerm(wv)}`);
            part += xv * wv;
          }
        }
        parts.push(part);
        lines.push({ lhs: `channel ${c}`, expr: terms.join(" + "), value: part, derived: true });
      }
      const bv = val(trace, bId, [o]);
      lines.push({ lhs: idxName(sym, index), expr: `${parts.map((v) => fmtTerm(v)).join(" + ")}${signed(bv)}`, value });
      return lines;
    }
    case "maxpool2d": {
      const [n, c, i, j] = index;
      const size = (node.attrs.size as number | undefined) ?? 2;
      const stride = (node.attrs.stride as number | undefined) ?? 2;
      const terms: string[] = [];
      for (let u = 0; u < size; u++) for (let v = 0; v < size; v++) terms.push(fmt(val(trace, ins[0], [n, c, i * stride + u, j * stride + v])));
      return [{ lhs: idxName(sym, index), expr: `max(${terms.join(", ")})`, value }];
    }
    case "flatten": {
      const xs = shape(ins[0]);
      const [n, k] = index;
      const rest = xs.slice(1);
      const src = [n];
      let rem = k;
      const tail: number[] = [];
      for (let a = rest.length - 1; a >= 0; a--) {
        tail.unshift(rem % rest[a]);
        rem = Math.floor(rem / rest[a]);
      }
      src.push(...tail);
      return [{ lhs: idxName(sym, index), expr: `${idxName(symbol(trace, ins[0]), src)} (same value, new position)`, value }];
    }
    case "bce": {
      const [pId, yId] = ins;
      const n = trace.tensor(pId).size;
      const terms: string[] = [];
      for (let i = 0; i < n; i++) {
        const p = val(trace, pId, [i]);
        const y = val(trace, yId, [i]);
        terms.push(`[${fmt(y)}·ln ${fmt(p)} + ${fmt(1 - y)}·ln ${fmt(1 - p)}]`);
      }
      return [{ lhs: "L", expr: `−(1/${n}) · (${sumTerms(terms)})`, value }];
    }
    case "mean":
    case "sum": {
      const t = trace.tensor(ins[0]);
      const axis = node.attrs.axis as number | undefined;
      if (axis !== undefined && t.values) {
        // Reduce one axis: the output index with the reduced axis re-inserted.
        const ax = (axis + t.shape.length) % t.shape.length;
        const terms = Array.from({ length: t.shape[ax] }, (_, k) => fmtTerm(val(trace, ins[0], [...index.slice(0, ax), k, ...index.slice(ax)])));
        const expr = node.op === "mean" ? `(${sumTerms(terms)}) / ${t.shape[ax]}` : sumTerms(terms);
        return [{ lhs: idxName(sym, index), expr, value }];
      }
      const terms = Array.from(t.values ?? []).map((v) => fmtTerm(v));
      const expr = node.op === "mean" ? `(${sumTerms(terms)}) / ${t.size}` : sumTerms(terms);
      return [{ lhs: node.op, expr, value }];
    }
    default:
      return null;
  }
}

function workedBackward(trace: TraceView, node: NodeRecord, outId: string, index: number[]): WorkedLine[] | null {
  const value = val(trace, outId, index);
  const gId = node.inputs[0];
  const fwdIns = node.inputs.slice(1);
  const k = node.outputs.indexOf(outId);
  const gradOf = node.attrs.grad_of?.[k];
  const shape = (t: string) => trace.tensor(t).shape;
  const name = gradOf ? symbol(trace, gradOf) : "x";
  const lhs = `∂L/∂${idxName(name, index)}`;
  const fwd = node.forward_of ? trace.node(node.forward_of) : null;
  switch (node.op) {
    case "grad_accumulate": {
      const terms = node.inputs.map((t) => fmtTerm(val(trace, t, index)));
      return [{ lhs, expr: sumTerms(terms), value }];
    }
    case "sigmoid": {
      if (!fwd) return null;
      const s = val(trace, fwd.outputs[0], index);
      const g = val(trace, gId, index);
      return [{ lhs, expr: `${fmtTerm(g)} · ${fmt(s)} · (1 − ${fmt(s)})`, value }];
    }
    case "tanh": {
      if (!fwd) return null;
      const t = val(trace, fwd.outputs[0], index);
      const g = val(trace, gId, index);
      return [{ lhs, expr: `${fmtTerm(g)} · (1 − ${fmt(t)}²)`, value }];
    }
    case "relu": {
      const z = val(trace, fwdIns[0], index);
      const g = val(trace, gId, index);
      return [{ lhs, expr: `${fmtTerm(g)} · [${fmt(z)} > 0]`, value }];
    }
    case "scale": {
      const g = val(trace, gId, index);
      return [{ lhs, expr: `${fmt(fwd?.attrs.c as number)} · ${fmtTerm(g)}`, value }];
    }
    case "transpose": {
      const axes = (fwd?.attrs.axes as number[]) ?? [];
      const src = axes.map((a) => index[a]);
      return [{ lhs, expr: `g${src.map(sub).join(",")} = ${fmt(val(trace, gId, src))}`, value }];
    }
    case "softmax": {
      // ∂L/∂x_i = s_i (g_i − Σ_j g_j s_j), over the softmax axis.
      if (!fwd) return null;
      const sId = fwd.outputs[0];
      const sT = trace.tensor(sId);
      const axis = (((fwd.attrs.axis as number | undefined) ?? -1) + sT.shape.length) % sT.shape.length;
      const n = sT.shape[axis];
      const at = (tid: string, kk: number) => val(trace, tid, index.map((d, a) => (a === axis ? kk : d)));
      let dot = 0;
      const terms: string[] = [];
      for (let kk = 0; kk < n; kk++) {
        dot += at(gId, kk) * at(sId, kk);
        terms.push(`${fmtTerm(at(gId, kk))}·${fmtTerm(at(sId, kk))}`);
      }
      const i0 = index[axis];
      return [
        { lhs: "Σ g·s", expr: sumTerms(terms), value: dot, derived: true },
        { lhs, expr: `${fmt(at(sId, i0))} · (${fmtTerm(at(gId, i0))} − ${fmtTerm(dot)})`, value },
      ];
    }
    case "maxpool2d": {
      // The gradient lands on the winner of each window; every other input cell gets 0.
      if (!fwd) return null;
      const [n, c, h, w] = index;
      const W = shape(outId)[3];
      const argmax = fwd.attrs.argmax as number[][][][];
      const flat = h * W + w;
      const hits: string[] = [];
      const rows = argmax[n][c];
      for (let i = 0; i < rows.length; i++) for (let j = 0; j < rows[i].length; j++) if (rows[i][j] === flat) hits.push(fmtTerm(val(trace, gId, [n, c, i, j])));
      const expr = hits.length ? `${hits.join(" + ")} (this cell won its window)` : "0 (this cell did not win its window)";
      return [{ lhs, expr, value }];
    }
    case "flatten": {
      const outShape = shape(outId);
      const rest = outShape.slice(1);
      let flat = 0;
      for (let a = 0; a < rest.length; a++) flat = flat * rest[a] + index[a + 1];
      return [{ lhs, expr: `g${sub(index[0])},${sub(flat)} = ${fmt(val(trace, gId, [index[0], flat]))} (same value, reshaped)`, value }];
    }
    case "bce": {
      const [pId, yId] = fwdIns;
      const n = trace.tensor(pId).size;
      const p = val(trace, pId, index);
      const y = val(trace, yId, index);
      const g = val(trace, gId, []);
      return [
        {
          lhs,
          expr: `${fmtTerm(g)} · (${fmt(p)} − ${fmt(y)}) / (${n} · ${fmt(p)} · (1 − ${fmt(p)}))`,
          value,
        },
      ];
    }
    case "linear":
    case "matmul": {
      const [xId, wId] = fwdIns;
      const xs = shape(xId);
      const ws = shape(wId);
      const wIs1d = ws.length === 1;
      if (gradOf === xId) {
        // ∂L/∂x[…, i, j] = Σ_c g[…, i, c] · w[…, j, c]
        const lead = index.slice(0, -2);
        const [i, j] = index.slice(-2);
        if (wIs1d) return [{ lhs, expr: `${fmtTerm(val(trace, gId, [i]))} · ${fmtTerm(val(trace, wId, [j]))}`, value }];
        const m = ws[ws.length - 1];
        const wLead = lead.slice(lead.length - (ws.length - 2));
        const terms = [];
        for (let c = 0; c < m; c++) terms.push(`${fmtTerm(val(trace, gId, [...lead, i, c]))}·${fmtTerm(val(trace, wId, [...wLead, j, c]))}`);
        return [{ lhs, expr: sumTerms(terms), value }];
      }
      if (gradOf === wId) {
        // ∂L/∂w[…, j, c] = Σ over the batch/positions of x[…, i, j] · g[…, i, c]
        const [j, c] = index.slice(-2);
        const wLead = index.slice(0, -2);
        const terms: string[] = [];
        const leadDims = xs.slice(0, -2);
        const count = leadDims.reduce((a, b) => a * b, 1);
        for (let flat = 0; flat < count; flat++) {
          const lead: number[] = [];
          let rem = flat;
          for (let a = leadDims.length - 1; a >= 0; a--) { lead.unshift(rem % leadDims[a]); rem = Math.floor(rem / leadDims[a]); }
          if (wLead.length && wLead.some((d, a) => d !== lead[lead.length - wLead.length + a])) continue;
          for (let i = 0; i < xs[xs.length - 2]; i++) {
            const g = val(trace, gId, wIs1d ? [...lead, i] : [...lead, i, c]);
            terms.push(`${fmtTerm(val(trace, xId, [...lead, i, j]))}·${fmtTerm(g)}`);
          }
        }
        return [{ lhs, expr: sumTerms(terms), value }];
      }
      // bias: g summed over every leading position
      const c = index[0] ?? 0;
      const gT = trace.tensor(gId);
      const terms: string[] = [];
      const cols = gT.shape[gT.shape.length - 1];
      for (let flat = 0; flat < gT.size; flat++) if (wIs1d || flat % cols === c) terms.push(fmtTerm(gT.values![flat]));
      return [{ lhs, expr: sumTerms(terms), value }];
    }
    case "add": {
      const g = trace.tensor(gId);
      if (g.size === trace.tensor(outId).size) {
        return [{ lhs, expr: fmt(val(trace, gId, index)), value }];
      }
      return [{ lhs, expr: `Σ g over broadcast axes (${g.size} values)`, value }];
    }
    case "mul": {
      const other = fwdIns[gradOf === fwdIns[0] ? 1 : 0];
      const gT = trace.tensor(gId);
      if (gT.size !== trace.tensor(outId).size) return null;
      const o = val(trace, other, bcastIndex(shape(other), index));
      return [{ lhs, expr: `${fmtTerm(val(trace, gId, index))} · ${fmtTerm(o)}`, value }];
    }
    case "mean": {
      const axis = fwd?.attrs.axis as number | undefined;
      if (axis !== undefined) {
        const t = trace.tensor(outId);
        const ax = (axis + t.shape.length) % t.shape.length;
        const gIdx = [...index.slice(0, ax), ...index.slice(ax + 1)];
        return [{ lhs, expr: `${fmtTerm(val(trace, gId, gIdx))} / ${t.shape[ax]}`, value }];
      }
      const n = trace.tensor(outId).size;
      return [{ lhs, expr: `${fmtTerm(val(trace, gId, []))} / ${n}`, value }];
    }
    case "sum":
      return [{ lhs, expr: fmt(val(trace, gId, [])), value }];
    default:
      return null;
  }
}
