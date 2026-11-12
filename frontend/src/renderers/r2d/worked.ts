/**
 * Worked arithmetic for one output cell of a node, built only from trace values.
 * Returns null when no rule exists for the op (the UI then shows values only).
 */
import type { TraceView } from "../../trace/load";
import type { NodeRecord } from "../../trace/types";
import { fmt, fmtTerm, sub } from "../../format";

export interface WorkedLine {
  /** Left-hand side, e.g. "z₀" or "∂L/∂w₁". */
  lhs: string;
  /** The expression with numbers substituted. */
  expr: string;
  /** The value of the output cell as stored in the trace. */
  value: number;
}

const MAX_TERMS = 12;

/** Symbols for unnamed tensors, by the op that produced them. */
const SYMBOL: Record<string, string> = { linear: "z", matmul: "z", sigmoid: "σ", tanh: "a", relu: "a", add: "c", mul: "c" };

function symbol(trace: TraceView, tid: string): string {
  const t = trace.tensor(tid);
  if (t.name) return t.name;
  const p = trace.producer(tid);
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

function workedForward(trace: TraceView, node: NodeRecord, outId: string, index: number[]): WorkedLine[] | null {
  const value = val(trace, outId, index);
  const ins = node.inputs;
  const shape = (t: string) => trace.tensor(t).shape;
  switch (node.op) {
    case "linear":
    case "matmul": {
      const [xId, wId] = ins;
      const bId = node.op === "linear" ? ins[2] : null;
      const [i, j] = index; // z (n,) or (n, m)
      const f = shape(xId)[1];
      const wIs1d = shape(wId).length === 1;
      const terms: string[] = [];
      for (let k = 0; k < f; k++) {
        const xv = val(trace, xId, [i, k]);
        const wv = val(trace, wId, wIs1d ? [k] : [k, j]);
        terms.push(`${fmtTerm(xv)}·${fmtTerm(wv)}`);
      }
      let expr = sumTerms(terms);
      if (bId) {
        const bShape = shape(bId);
        const bv = val(trace, bId, bShape.length === 0 ? [] : [wIs1d ? 0 : j]);
        expr += ` + ${fmtTerm(bv)}`;
      }
      return [{ lhs: idxName("z", index), expr, value }];
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
      const sym = node.op === "add" ? "+" : "·";
      return [{ lhs: idxName("c", index), expr: `${fmtTerm(a)} ${sym} ${fmtTerm(b)}`, value }];
    }
    case "bce": {
      const [pId, yId] = ins;
      const n = trace.tensor(pId).size;
      const terms: string[] = [];
      for (let i = 0; i < n; i++) {
        const p = val(trace, pId, [i]);
        const y = val(trace, yId, [i]);
        const term = y * Math.log(Math.max(p, 1e-12)) + (1 - y) * Math.log(Math.max(1 - p, 1e-12));
        terms.push(`[${fmt(y)}·ln ${fmt(p)} + ${fmt(1 - y)}·ln ${fmt(1 - p)}]`);
        void term;
      }
      return [{ lhs: "L", expr: `−(1/${n}) · (${sumTerms(terms)})`, value }];
    }
    case "mean":
    case "sum": {
      const t = trace.tensor(ins[0]);
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
  switch (node.op) {
    case "grad_accumulate": {
      const terms = node.inputs.map((t) => fmtTerm(val(trace, t, index)));
      return [{ lhs, expr: sumTerms(terms), value }];
    }
    case "sigmoid": {
      const fwd = node.forward_of ? trace.node(node.forward_of) : null;
      if (!fwd) return null;
      const s = val(trace, fwd.outputs[0], index);
      const g = val(trace, gId, index);
      return [{ lhs, expr: `${fmtTerm(g)} · ${fmt(s)} · (1 − ${fmt(s)})`, value }];
    }
    case "tanh": {
      const fwd = node.forward_of ? trace.node(node.forward_of) : null;
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
      const wIs1d = shape(wId).length === 1;
      const n = shape(xId)[0];
      const f = shape(xId)[1];
      if (gradOf === xId) {
        const [i, j] = index;
        if (wIs1d) {
          return [{ lhs, expr: `${fmtTerm(val(trace, gId, [i]))} · ${fmtTerm(val(trace, wId, [j]))}`, value }];
        }
        const m = shape(wId)[1];
        const terms = [];
        for (let c = 0; c < m; c++) terms.push(`${fmtTerm(val(trace, gId, [i, c]))}·${fmtTerm(val(trace, wId, [j, c]))}`);
        return [{ lhs, expr: sumTerms(terms), value }];
      }
      if (gradOf === wId) {
        const [j, c] = index;
        const terms = [];
        for (let i = 0; i < n; i++) {
          const g = val(trace, gId, wIs1d ? [i] : [i, c]);
          terms.push(`${fmtTerm(val(trace, xId, [i, j]))}·${fmtTerm(g)}`);
        }
        return [{ lhs, expr: sumTerms(terms), value }];
      }
      // bias: sum of g over samples
      const c = index[0] ?? 0;
      const terms = [];
      for (let i = 0; i < n; i++) terms.push(fmtTerm(val(trace, gId, wIs1d ? [i] : [i, c])));
      void f;
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
      const n = trace.tensor(outId).size;
      return [{ lhs, expr: `${fmtTerm(val(trace, gId, []))} / ${n}`, value }];
    }
    case "sum":
      return [{ lhs, expr: fmt(val(trace, gId, [])), value }];
    default:
      return null;
  }
}
