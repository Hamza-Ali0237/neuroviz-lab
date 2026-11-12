import { useAppState, useDispatch } from "../../state/store";
import type { EventRecord, NodeRecord } from "../../trace/types";
import { fmt, fmtTerm, sub, shapeStr } from "../../format";
import { unflatIndex } from "../../trace/load";
import { MatrixView } from "./MatrixView";
import { OpCard } from "./OpCard";
import { LRScene } from "./LRScene";
import { worked } from "./worked";

/** Worked computation for the selected (or first) output cell of a node. */
function Worked({ node }: { node: NodeRecord }) {
  const { trace, selection, sampleIndex } = useAppState();
  const selectedHere = selection.tensorId && selection.index && node.outputs.includes(selection.tensorId);
  const outId = selectedHere ? selection.tensorId! : node.outputs[0];
  if (!outId) return null;
  const t = trace.tensor(outId);
  const defaultIndex = t.shape.map((d, i) => (i === 0 && d === trace.raw.mode.n_samples && t.role !== "param" ? sampleIndex : 0));
  const index = selectedHere ? selection.index! : defaultIndex;
  const lines = worked(trace, node, outId, index);
  const where = `${trace.label(outId)}[${index.join(", ")}]`;
  return (
    <div className="nvz-worked">
      <h4>
        Worked computation <span className="nvz-muted">for {where}</span>
        {!selectedHere && <span className="nvz-muted"> · click any output cell to change</span>}
      </h4>
      {lines ? (
        lines.map((l, i) => (
          <p key={i} className="mono nvz-worked-line">
            <span className="lhs">{l.lhs}</span> = {l.expr} = <b>{fmt(l.value)}</b>
          </p>
        ))
      ) : (
        <p className="nvz-muted">No worked arithmetic rule for “{node.op}”; the values above are from the trace.</p>
      )}
    </div>
  );
}

function NodeView({ node }: { node: NodeRecord }) {
  const { trace, selection } = useAppState();
  const isBackward = node.phase === "backward";
  const gradOf = node.attrs.grad_of ?? [];
  return (
    <article className={`nvz-node ${selection.nodeId === node.id ? "selected" : ""}`}>
      <OpCard node={node} />
      <Worked node={node} />
      <div className="nvz-tensors">
        <div className="nvz-tensor-group">
          <h4>{isBackward ? "Inputs (g first, then the forward inputs)" : "Inputs"}</h4>
          {node.inputs.map((tid) => (
            <MatrixView key={tid} tensorId={tid} />
          ))}
        </div>
        <div className="nvz-tensor-group">
          <h4>{isBackward ? "Gradients" : "Output"}</h4>
          {node.outputs.map((tid, k) => (
            <MatrixView key={tid} tensorId={tid} caption={isBackward && gradOf[k] ? `∂L/∂${trace.label(gradOf[k])}` : undefined} />
          ))}
        </div>
      </div>
    </article>
  );
}

function UpdateEvent({ e }: { e: EventRecord }) {
  const { trace, selection } = useAppState();
  const p = e.payload as { param: string; lr: number; before: string; grad: string; after: string };
  const before = trace.tensor(p.before);
  const after = trace.tensor(p.after);
  const grad = trace.tensor(p.grad);
  const flatSel = selection.tensorId === p.after || selection.tensorId === p.before || selection.tensorId === p.grad ? selection.index : null;
  const shown = flatSel ? [flatSel] : Array.from({ length: Math.min(before.size, 8) }, (_, i) => unflatIndex(before.shape, i));
  return (
    <article className="nvz-node nvz-update">
      <h3>
        {p.param} ← {p.param} − lr · ∂L/∂{p.param} <span className="nvz-muted">(lr = {p.lr})</span>
      </h3>
      <div className="nvz-tensors">
        <MatrixView tensorId={p.before} caption={`${p.param} before`} />
        <MatrixView tensorId={p.grad} caption={`∂L/∂${p.param}`} />
        <MatrixView tensorId={p.after} caption={`${p.param} after`} />
      </div>
      <div className="nvz-worked">
        <h4>Worked computation</h4>
        {shown.map((idx) => {
          const b = trace.value(p.before, idx);
          const g = trace.value(p.grad, idx);
          const a = trace.value(p.after, idx);
          const nm = `${p.param}${idx.map(sub).join(",")}`;
          return (
            <p key={idx.join(",")} className="mono nvz-worked-line">
              <span className="lhs">{nm}′</span> = {fmtTerm(b)} − {p.lr}·{fmtTerm(g)} = <b>{fmt(a)}</b>
            </p>
          );
        })}
        {before.size > shown.length && <p className="nvz-muted">… {before.size - shown.length} more; click a cell to see its line.</p>}
      </div>
      <p className="nvz-muted">
        {p.param}: {shapeStr(before.shape)} · mean {fmt(before.stats.mean)} → {fmt(after.stats.mean)} · grad mean {fmt(grad.stats.mean)}
      </p>
    </article>
  );
}

function PredictionEvent({ e }: { e: EventRecord }) {
  const { trace, sampleIndex } = useAppState();
  const dispatch = useDispatch();
  const p = e.payload as { threshold: number; predicted_class: number[] };
  const probs = trace.tensor(e.refs[0]);
  const y = [...trace.tensors.values()].find((t) => t.role === "label");
  const classes = trace.raw.dataset?.classes;
  return (
    <article className="nvz-node">
      <h3>Prediction <span className="nvz-muted">class = [p &gt; {p.threshold}]</span></h3>
      <table className="nvz-table mono">
        <thead>
          <tr><th>sample</th><th>p</th><th>predicted</th>{y && <th>label</th>}</tr>
        </thead>
        <tbody>
          {p.predicted_class.map((c, i) => (
            <tr key={i} className={i === sampleIndex ? "nvz-row-sample" : ""} onClick={() => dispatch({ type: "sample", index: i })}>
              <td>{i}</td>
              <td title={`${trace.label(probs.id)}[${i}] = ${probs.values?.[i]}`}>{fmt(probs.values?.[i])}</td>
              <td>{classes ? classes[c] : c}</td>
              {y && <td>{classes ? classes[Math.round(y.values?.[i] ?? 0)] : fmt(y.values?.[i])}</td>}
            </tr>
          ))}
        </tbody>
      </table>
      <MatrixView tensorId={probs.id} />
    </article>
  );
}

function GenericEvent({ e }: { e: EventRecord }) {
  const { trace } = useAppState();
  return (
    <article className="nvz-node">
      <h3>{e.kind.replace(/_/g, " ")}</h3>
      <dl className="nvz-payload mono">
        {Object.entries(e.payload).map(([k, v]) => (
          <div key={k}><dt>{k}</dt><dd>{typeof v === "string" && trace.tensors.has(v) ? `${v} (${trace.label(v)})` : JSON.stringify(v)}</dd></div>
        ))}
      </dl>
      {e.refs.map((tid) => (
        <MatrixView key={tid} tensorId={tid} />
      ))}
    </article>
  );
}

function EventView({ e }: { e: EventRecord }) {
  if (e.kind === "sgd_update") return <UpdateEvent e={e} />;
  if (e.kind === "prediction") return <PredictionEvent e={e} />;
  return <GenericEvent e={e} />;
}

export function Scene2D() {
  const { trace, stepIndex } = useAppState();
  const step = trace.steps[stepIndex];
  const nodes = trace.nodesInStep(stepIndex);
  const events = trace.eventsInStep(stepIndex);
  const hasScene = trace.raw.model.family === "linear" && [1, 2].includes(trace.raw.mode.n_features);
  return (
    <div className="nvz-scene2d">
      <div className="nvz-flow">
        <header className="nvz-stephead">
          <span className={`nvz-phase-pill nvz-phase-${step.phase}`}>{step.phase}</span>
          <h2>{step.title}</h2>
          {trace.raw.kind === "train" && <span className="nvz-muted">iteration {step.iteration + 1}</span>}
        </header>
        {nodes.map((n) => (
          <NodeView key={n.id} node={n} />
        ))}
        {events.map((e) => (
          <EventView key={e.id} e={e} />
        ))}
        {!nodes.length && !events.length && <p className="nvz-muted">Nothing recorded in this step.</p>}
      </div>
      {hasScene && (
        <div className="nvz-side">
          <LRScene />
        </div>
      )}
    </div>
  );
}
