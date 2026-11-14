import { useAppState, useDispatch } from "../../state/store";
import type { EventRecord, NodeRecord } from "../../trace/types";
import { eventLabel, fmt, fmtTerm, keyLabel, sub, shapeStr } from "../../format";
import { unflatIndex } from "../../trace/load";
import { MatrixView } from "./MatrixView";
import { OpCard } from "./OpCard";
import { Worked } from "./WorkedView";
import { LRScene } from "./LRScene";
import { MLPScene } from "./MLPScene";
import { TreeScene } from "./TreeScene";
import { CNNScene, cnnNode } from "./CNNScene";
import { TransformerScene, attentionNode } from "./TransformerScene";
import { treeAt } from "../tree";
import { mlpView } from "../mlp";

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
      <h3>{eventLabel(e.kind)}</h3>
      <dl className="nvz-payload mono">
        {Object.entries(e.payload).map(([k, v]) => (
          <div key={k}><dt>{keyLabel(k)}</dt><dd>{typeof v === "string" && trace.tensors.has(v) ? `${v} (${trace.label(v)})` : JSON.stringify(v)}</dd></div>
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

/** The model-specific plot for this trace, or null when there is none (generic traces). */
function ModelScene() {
  const { trace } = useAppState();
  if (trace.raw.model.family === "linear" && [1, 2].includes(trace.raw.mode.n_features)) return <LRScene />;
  if (trace.raw.model.family === "mlp") return <MLPScene />;
  if (trace.raw.model.family === "tree") return <TreeScene />;
  if (trace.raw.model.family === "cnn") return <CNNScene />;
  if (trace.raw.model.family === "transformer") return <TransformerScene />;
  return null;
}

/** Whether the model scene has something to draw at this step (capture mode falls back to the ops otherwise). */
function sceneAt(trace: ReturnType<typeof useAppState>["trace"], stepIndex: number): boolean {
  const fam = trace.raw.model.family;
  if (fam === "cnn") return cnnNode(trace, stepIndex) !== null;
  if (fam === "transformer") {
    const attn = attentionNode(trace);
    return trace.nodesInStep(stepIndex).some((n) => n.op === "embedding" && n.phase === "forward") || (!!attn && attn.step <= stepIndex);
  }
  return true;
}

/** Whether the trace has a model-specific 2D scene. */
export function hasModelScene(trace: ReturnType<typeof useAppState>["trace"]): boolean {
  if (trace.raw.model.family === "linear") return [1, 2].includes(trace.raw.mode.n_features);
  if (trace.raw.model.family === "mlp") return mlpView(trace, 0, 0) !== null;
  if (trace.raw.model.family === "tree") return treeAt(trace, trace.steps.length - 1) !== null;
  if (trace.raw.model.family === "cnn") return trace.raw.nodes.some((n) => n.op === "conv2d");
  if (trace.raw.model.family === "transformer") return attentionNode(trace) !== null;
  return false;
}

export function Scene2D() {
  const { trace, stepIndex, capture } = useAppState();
  const step = trace.steps[stepIndex];
  const nodes = trace.nodesInStep(stepIndex);
  const events = trace.eventsInStep(stepIndex);
  const hasScene = hasModelScene(trace) && sceneAt(trace, stepIndex);
  if (capture) {
    // Capture mode: the scene alone, large; the flow of ops only when the model has no scene.
    return (
      <div className="nvz-scene2d nvz-capture-2d">
        {hasScene ? <ModelScene /> : nodes.map((n) => <NodeView key={n.id} node={n} />)}
        {!hasScene && events.map((e) => <EventView key={e.id} e={e} />)}
      </div>
    );
  }
  // Forward steps: the model's plot comes first so it is visible without scrolling.
  const plotFirst = hasScene && (step.phase === "forward" || ["tree", "cnn", "transformer"].includes(trace.raw.model.family));
  return (
    <div className={`nvz-scene2d ${plotFirst ? "nvz-plot-first" : ""}`}>
      <div className="nvz-flow">
        <header className="nvz-stephead">
          <span className={`nvz-phase-pill nvz-phase-${step.phase}`}>{step.phase}</span>
          <h2>{step.title}</h2>
          {trace.raw.kind === "train" && <span className="nvz-muted">iteration {step.iteration + 1}</span>}
        </header>
        {plotFirst && <ModelScene />}
        {nodes.map((n) => (
          <NodeView key={n.id} node={n} />
        ))}
        {events.map((e) => (
          <EventView key={e.id} e={e} />
        ))}
        {!nodes.length && !events.length && <p className="nvz-muted">Nothing recorded in this step.</p>}
      </div>
      {hasScene && !plotFirst && (
        <div className="nvz-side">
          <ModelScene />
        </div>
      )}
    </div>
  );
}
