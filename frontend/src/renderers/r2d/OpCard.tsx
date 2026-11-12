import katex from "katex";
import { useAppState, useDispatch } from "../../state/store";
import type { NodeRecord } from "../../trace/types";
import { shapeStr } from "../../format";

export function Formula({ tex }: { tex: string }) {
  const html = katex.renderToString(tex, { throwOnError: false, displayMode: true });
  return <div className="nvz-formula" dangerouslySetInnerHTML={{ __html: html }} />;
}

/** Input shapes → op → output shapes; tensors are clickable. */
export function ShapeFlow({ node }: { node: NodeRecord }) {
  const { trace } = useAppState();
  const dispatch = useDispatch();
  const chip = (tid: string) => (
    <button
      key={tid}
      className="nvz-chip mono"
      title={`${tid} · ${trace.tensor(tid).role}`}
      onClick={() => dispatch({ type: "select", selection: { tensorId: tid } })}
    >
      {trace.label(tid)} {shapeStr(trace.tensor(tid).shape)}
    </button>
  );
  return (
    <div className="nvz-shapeflow">
      <span className="nvz-chips">{node.inputs.map(chip)}</span>
      <span className="nvz-arrow">→</span>
      <span className={`nvz-opname nvz-phase-${node.phase}`}>{node.op}</span>
      <span className="nvz-arrow">→</span>
      <span className="nvz-chips">{node.outputs.map(chip)}</span>
    </div>
  );
}

/** Op name, KaTeX formula, text, shape rule and tags. Opaque ops get a clear fallback. */
export function OpCard({ node }: { node: NodeRecord }) {
  const { trace } = useAppState();
  const dispatch = useDispatch();
  const ex = node.explain;
  const fwd = node.forward_of ? trace.node(node.forward_of) : null;
  return (
    <section className={`nvz-opcard nvz-phase-${node.phase}`}>
      <header>
        <h3>
          <button className="nvz-link" onClick={() => dispatch({ type: "select", selection: { nodeId: node.id } })} title="Inspect this node">
            {ex?.title ?? node.op}
          </button>
        </h3>
        <span className="nvz-tag">{node.category}</span>
        {node.origin === "user" && <span className="nvz-tag nvz-tag-user">custom op</span>}
        {!node.differentiable && <span className="nvz-tag nvz-tag-warn">non-differentiable</span>}
        <span className="nvz-muted">{node.id}</span>
      </header>
      {fwd && (
        <p className="nvz-backref">
          backward of{" "}
          <button
            className="nvz-link"
            onClick={() => {
              dispatch({ type: "goto", step: fwd.step });
              dispatch({ type: "select", selection: { nodeId: fwd.id } });
            }}
          >
            {fwd.op} ({fwd.id}, step {fwd.step})
          </button>
          ; g is the upstream gradient
        </p>
      )}
      {ex ? (
        <>
          {ex.formula && <Formula tex={ex.formula} />}
          {ex.text && <p className="nvz-text">{ex.text}</p>}
          {ex.shape_rule && (
            <p className="nvz-muted mono nvz-shape-rule">shapes: {ex.shape_rule}</p>
          )}
        </>
      ) : (
        <p className="nvz-opaque">No explanation available: values only.</p>
      )}
      <ShapeFlow node={node} />
    </section>
  );
}
