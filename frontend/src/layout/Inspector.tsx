import { useAppState, useDispatch } from "../state/store";
import { fmt, fmt6, fmtFull, shapeStr } from "../format";
import { unflatIndex } from "../trace/load";

export function Inspector() {
  const { trace, selection } = useAppState();
  const dispatch = useDispatch();
  const node = selection.nodeId ? trace.node(selection.nodeId) : null;
  const tensor = selection.tensorId ? trace.tensor(selection.tensorId) : null;
  if (!tensor && !node) {
    return (
      <aside className="nvz-inspector nvz-inspector-rail" title="Click a number or a tensor name to inspect it. Hover any number for its source.">
        <h2>Inspector</h2>
        <p className="nvz-muted">Click a number or a tensor name to inspect it.</p>
      </aside>
    );
  }
  const producer = tensor ? trace.producer(tensor.id) : null;
  const sel = selection.index;
  return (
    <aside className="nvz-inspector">
      <h2>
        Inspector{" "}
        <button className="nvz-link" onClick={() => dispatch({ type: "clearSelection" })} title="Clear selection (Esc)">
          clear
        </button>
      </h2>
      {node && !tensor && (
        <dl>
          <dt>node</dt>
          <dd>
            {node.id} · {node.op} <span className="nvz-tag">{node.category}</span>
          </dd>
          <dt>phase</dt>
          <dd>{node.phase}</dd>
          <dt>inputs</dt>
          <dd>{node.inputs.map((t) => `${trace.label(t)} ${shapeStr(trace.tensor(t).shape)}`).join(", ")}</dd>
          <dt>outputs</dt>
          <dd>{node.outputs.map((t) => `${trace.label(t)} ${shapeStr(trace.tensor(t).shape)}`).join(", ")}</dd>
        </dl>
      )}
      {tensor && (
        <>
          <dl>
            <dt>tensor</dt>
            <dd>
              <b>{trace.label(tensor.id)}</b> <span className="nvz-muted">{tensor.id}</span>
            </dd>
            <dt>role</dt>
            <dd>
              <span className="nvz-tag">{tensor.role}</span>
            </dd>
            <dt>shape</dt>
            <dd className="mono">
              {shapeStr(tensor.shape)} · {tensor.size} {tensor.size === 1 ? "value" : "values"}
            </dd>
            <dt>stats</dt>
            <dd className="mono">
              min {fmt(tensor.stats.min)} · max {fmt(tensor.stats.max)} · mean {fmt(tensor.stats.mean)}
            </dd>
            <dt>produced by</dt>
            <dd>
              {producer ? (
                <button className="nvz-link" onClick={() => dispatch({ type: "goto", step: producer.step })}>
                  {producer.op} ({producer.id}, step {producer.step})
                </button>
              ) : (
                <span className="nvz-muted">leaf ({tensor.role})</span>
              )}
            </dd>
            {sel && (
              <>
                <dt>selected</dt>
                <dd className="mono" title={`${trace.label(tensor.id)}[${sel.join(", ")}] = ${fmtFull(trace.value(tensor.id, sel))}`}>
                  {trace.label(tensor.id)}[{sel.join(", ")}] = {fmt6(trace.value(tensor.id, sel))}
                </dd>
              </>
            )}
          </dl>
          <h3>Values</h3>
          {tensor.values ? (
            <ol className="nvz-values mono">
              {Array.from(tensor.values).map((v, i) => {
                const idx = unflatIndex(tensor.shape, i);
                const active = sel && sel.every((d, k) => d === idx[k]);
                return (
                  <li
                    key={i}
                    className={active ? "active" : ""}
                    title={`${trace.label(tensor.id)}[${idx.join(", ")}] = ${fmtFull(v)}`}
                    onClick={() => dispatch({ type: "select", selection: { tensorId: tensor.id, index: idx } })}
                  >
                    <span className="nvz-idx">[{idx.join(", ")}]</span> {fmt6(v)}
                  </li>
                );
              })}
            </ol>
          ) : (
            <p className="nvz-muted">Values were truncated in the trace (too large); only stats are available.</p>
          )}
        </>
      )}
    </aside>
  );
}
