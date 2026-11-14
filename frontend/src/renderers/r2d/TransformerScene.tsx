import { useAppState, useDispatch } from "../../state/store";
import type { NodeRecord } from "../../trace/types";
import { POS_RGB, fmt, fmtFull } from "../../format";
import type { TraceView } from "../../trace/load";
import { tokenLabels, vocabOf } from "../tokens";

/** The attention-weights softmax node (forward, output named "attention weights"), or null. */
export function attentionNode(trace: TraceView): NodeRecord | null {
  return trace.raw.nodes.find((n) => n.phase === "forward" && n.op === "softmax" && trace.tensor(n.outputs[0]).name === "attention weights") ?? null;
}

/**
 * The T×T attention weights for the selected sample with token labels on both axes
 * (row = the token that is looking, column = where it looks). Click a row for its worked
 * softmax: the scaled scores, their exponentials, the sum and the weights.
 */
export function AttentionMap({ node }: { node: NodeRecord }) {
  const { trace, selection, sampleIndex, stepIndex, capture } = useAppState();
  const dispatch = useDispatch();
  const aId = node.outputs[0];
  const sId = node.inputs[0];
  const a = trace.tensor(aId);
  const [nS, T] = a.shape;
  const n = Math.min(sampleIndex, nS - 1);
  const tokens = tokenLabels(trace, n) ?? Array.from({ length: T }, (_, i) => `t${i}`);
  const at = (tid: string, i: number, j: number) => trace.value(tid, [n, i, j]) ?? 0;
  // Capture mode has no clicks: it shows the first row's softmax by default.
  const row = selection.tensorId === aId && selection.index?.[0] === n ? selection.index[1] : capture ? 0 : null;
  const scores = row === null ? null : Array.from({ length: T }, (_, j) => at(sId, row, j));
  const exps = scores?.map(Math.exp) ?? null;
  const total = exps?.reduce((x, y) => x + y, 0) ?? 0;
  const weights = row === null ? null : Array.from({ length: T }, (_, j) => at(aId, row, j));
  const wsum = weights?.reduce((x, y) => x + y, 0) ?? 0;
  const computed = node.step <= stepIndex;
  return (
    <figure className="nvz-scene nvz-attention">
      <figcaption>
        Attention weights, sample {n}{" "}
        <span className="nvz-muted">row = the token that is looking, column = where it looks; each row sums to 1. Click a row for its softmax.{computed ? "" : " (computed at step " + node.step + ")"}</span>
      </figcaption>
      <table className="nvz-attn mono">
        <thead>
          <tr><th />{tokens.map((t, j) => <th key={j} scope="col">{t}</th>)}</tr>
        </thead>
        <tbody>
          {tokens.map((t, i) => (
            <tr key={i} className={i === row ? "selected" : ""} onClick={() => dispatch({ type: "select", selection: { tensorId: aId, index: [n, i, 0] } })}>
              <th scope="row">{t}</th>
              {tokens.map((_, j) => {
                const v = at(aId, i, j);
                return (
                  <td key={j} className="nvz-cell" style={{ "--tint": `rgba(${POS_RGB.join(",")}, ${(0.08 + 0.8 * v).toFixed(3)})` } as React.CSSProperties} title={`attention weights[${n}, ${i}, ${j}] = ${fmtFull(v)} (${tokens[i]} → ${tokens[j]})`}>
                    {fmt(v)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {row !== null && scores && exps && weights && (
        <div className="nvz-worked">
          <h4>Worked softmax <span className="nvz-muted">for row {tokens[row]} (what {tokens[row]} attends to)</span></h4>
          <table className="nvz-table mono nvz-softmax-row">
            <thead><tr><th /> {tokens.map((t, j) => <th key={j}>{t}</th>)}</tr></thead>
            <tbody>
              <tr><th>scaled score s</th>{scores.map((v, j) => <td key={j} title={`${trace.label(sId)}[${n}, ${row}, ${j}] = ${fmtFull(v)}`}>{fmt(v)}</td>)}</tr>
              <tr><th>eˢ</th>{exps.map((v, j) => <td key={j} title="computed here from the recorded score">{fmt(v)}</td>)}</tr>
              <tr><th>weight = eˢ / Σ</th>{weights.map((v, j) => <td key={j} title={`attention weights[${n}, ${row}, ${j}] = ${fmtFull(v)} (trace)`}><b>{fmt(v)}</b></td>)}</tr>
            </tbody>
          </table>
          <p className="mono nvz-worked-line">Σ eˢ = {exps.map((v) => fmt(v)).join(" + ")} = <b>{fmt(total)}</b> <span className="nvz-muted">(computed here)</span></p>
          <p className="mono nvz-worked-line">Σ weights = <b>{fmt(wsum)}</b> <span className="nvz-muted">(from the trace; a softmax row sums to 1)</span></p>
        </div>
      )}
    </figure>
  );
}

/** Embedding step: which table row each token picked (token → row index → vector). */
export function EmbeddingPick({ node }: { node: NodeRecord }) {
  const { trace, sampleIndex, selection } = useAppState();
  const dispatch = useDispatch();
  const [idsId, tableId] = node.inputs;
  const ids = trace.tensor(idsId);
  const table = trace.tensor(tableId);
  const [nS, T] = ids.shape;
  const n = Math.min(sampleIndex, nS - 1);
  const tokens = tokenLabels(trace, n);
  const vocab = vocabOf(trace);
  const d = table.shape[1];
  if (!ids.values || !table.values) return null;
  const selRow = selection.tensorId === tableId && selection.index ? selection.index[0] : -1;
  return (
    <figure className="nvz-scene nvz-embedding">
      <figcaption>Token → row of {trace.label(tableId)} → vector <span className="nvz-muted">(sample {n}; the table has {table.shape[0]} rows, one per vocabulary entry)</span></figcaption>
      <table className="nvz-table mono">
        <thead><tr><th>position</th><th>token</th><th>id = row</th>{Array.from({ length: d }, (_, k) => <th key={k}>{trace.label(tableId)}[·, {k}]</th>)}</tr></thead>
        <tbody>
          {Array.from({ length: T }, (_, t) => {
            const id = Math.round(ids.values![n * T + t]);
            return (
              <tr key={t} className={id === selRow ? "selected" : ""} onClick={() => dispatch({ type: "select", selection: { tensorId: tableId, index: [id, 0] } })}>
                <td>{t}</td>
                <td>{tokens?.[t] ?? vocab?.[id] ?? id}</td>
                <td>{id}</td>
                {Array.from({ length: d }, (_, k) => (
                  <td key={k} title={`${trace.label(tableId)}[${id}, ${k}] = ${fmtFull(table.values![id * d + k])}`}>{fmt(table.values![id * d + k])}</td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </figure>
  );
}

/** Transformer model scene: the attention map (and, on the embedding step, the row picks). */
export function TransformerScene() {
  const { trace, stepIndex } = useAppState();
  const attn = attentionNode(trace);
  const emb = trace.nodesInStep(stepIndex).find((nd) => nd.op === "embedding" && nd.phase === "forward");
  if (emb) return <EmbeddingPick node={emb} />;
  if (attn && attn.step <= stepIndex) return <AttentionMap node={attn} />;
  return null;
}
