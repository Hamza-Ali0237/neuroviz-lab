import { useAppState, useDispatch } from "../../state/store";
import { fmt } from "../../format";
import type { Tensor } from "../../trace/load";

/**
 * Logistic-regression scene. 2 features: scatter of the samples with the decision
 * boundary w·x + b = 0 from the parameters current at this step. 1 feature: x vs
 * predicted probability with the sigmoid curve σ(w x + b). At an update step the
 * previous boundary/curve is dashed so the move is visible.
 */
export function LRScene() {
  const { trace, stepIndex, sampleIndex, selection } = useAppState();
  const dispatch = useDispatch();
  const x = [...trace.tensors.values()].find((t) => t.role === "input" && t.name === "x");
  const y = [...trace.tensors.values()].find((t) => t.role === "label");
  const params = trace.paramsAt(stepIndex);
  const w = params.get("w");
  const b = params.get("b");
  if (!x?.values || !w?.values || !b?.values) return null;
  const step = trace.steps[stepIndex];
  const prev = step.phase === "update" ? trace.paramsAt(stepIndex - 1) : null;
  const prevW = prev?.get("w");
  const prevB = prev?.get("b");
  const nf = trace.raw.mode.n_features;
  const n = trace.raw.mode.n_samples;
  const W = 420;
  const H = 300;
  const pad = 36;
  const label = (i: number) => (y?.values ? y.values[i] : null);
  const selRow = selection.index && selection.tensorId !== w.id && selection.tensorId !== b.id ? selection.index[0] : -1;
  const classes = trace.raw.dataset?.classes ?? ["0", "1"];
  const pointClass = (i: number) => {
    const l = label(i);
    const cls = l === null ? "nvz-pt-unlabelled" : l > 0.5 ? "nvz-pt-1" : "nvz-pt-0";
    return `nvz-pt ${cls} ${i === sampleIndex ? "current" : ""} ${i === selRow ? "selected" : ""}`;
  };
  const onPoint = (i: number) => {
    dispatch({ type: "sample", index: i });
    dispatch({ type: "select", selection: { tensorId: x.id, index: nf === 1 ? [i, 0] : [i, 0] } });
  };
  const title = (i: number) => {
    const l = label(i);
    const coords = Array.from({ length: nf }, (_, j) => fmt(x.values![i * nf + j])).join(", ");
    return `sample ${i}: x = (${coords})${l === null ? "" : `, y = ${fmt(l)}`}`;
  };

  if (nf === 2) {
    const xs = Array.from({ length: n }, (_, i) => x.values![i * 2]);
    const ys = Array.from({ length: n }, (_, i) => x.values![i * 2 + 1]);
    const lo = Math.min(-2.5, ...xs, ...ys) - 0.5;
    const hi = Math.max(2.5, ...xs, ...ys) + 0.5;
    const sx = (v: number) => pad + ((v - lo) / (hi - lo)) * (W - 2 * pad);
    const sy = (v: number) => H - pad - ((v - lo) / (hi - lo)) * (H - 2 * pad);
    const boundary = (wt: Tensor, bt: Tensor, cls: string) => {
      const [w1, w2] = wt.values!;
      const b0 = bt.values![0];
      // w1·x1 + w2·x2 + b = 0
      if (Math.abs(w2) < 1e-9 && Math.abs(w1) < 1e-9) return null;
      const pts =
        Math.abs(w2) >= Math.abs(w1)
          ? [lo, hi].map((x1) => [x1, -(w1 * x1 + b0) / w2])
          : [lo, hi].map((x2) => [-(w2 * x2 + b0) / w1, x2]);
      return (
        <line
          className={cls}
          x1={sx(pts[0][0])}
          y1={sy(pts[0][1])}
          x2={sx(pts[1][0])}
          y2={sy(pts[1][1])}
        />
      );
    };
    const names = trace.raw.dataset?.features ?? ["x1", "x2"];
    return (
      <figure className="nvz-scene">
        <figcaption>
          Decision boundary <span className="mono">{fmt(w.values[0])}·{names[0]} + {fmt(w.values[1])}·{names[1]} + {fmt(b.values[0])} = 0</span>
        </figcaption>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Samples and decision boundary">
          <line className="nvz-axis" x1={sx(lo)} y1={sy(0)} x2={sx(hi)} y2={sy(0)} />
          <line className="nvz-axis" x1={sx(0)} y1={sy(lo)} x2={sx(0)} y2={sy(hi)} />
          <text className="nvz-axis-label" x={W - pad} y={sy(0) - 6}>{names[0]}</text>
          <text className="nvz-axis-label" x={sx(0) + 6} y={pad}>{names[1]}</text>
          {prevW && prevB && boundary(prevW, prevB, "nvz-boundary prev")}
          {boundary(w, b, "nvz-boundary")}
          {xs.map((px, i) => (
            <circle key={i} className={pointClass(i)} cx={sx(px)} cy={sy(ys[i])} r={7} onClick={() => onPoint(i)}>
              <title>{title(i)}</title>
            </circle>
          ))}
        </svg>
        <SceneLegend classes={classes} labelled={!!y} prev={!!prevW} />
      </figure>
    );
  }

  // 1 feature: x vs probability with the sigmoid curve.
  const xs = Array.from({ length: n }, (_, i) => x.values![i]);
  const lo = Math.min(-4, ...xs) - 0.5;
  const hi = Math.max(4, ...xs) + 0.5;
  const sx = (v: number) => pad + ((v - lo) / (hi - lo)) * (W - 2 * pad);
  const sy = (p: number) => H - pad - p * (H - 2 * pad);
  const curve = (wt: Tensor, bt: Tensor, cls: string) => {
    const w0 = wt.values![0];
    const b0 = bt.values![0];
    const pts: string[] = [];
    for (let k = 0; k <= 80; k++) {
      const xv = lo + ((hi - lo) * k) / 80;
      pts.push(`${sx(xv).toFixed(1)},${sy(1 / (1 + Math.exp(-(w0 * xv + b0)))).toFixed(1)}`);
    }
    return <polyline className={cls} points={pts.join(" ")} />;
  };
  const out = trace.nodesInStep(stepIndex).find((nd) => nd.op === "sigmoid" && nd.phase === "forward");
  const probs = out ? trace.tensor(out.outputs[0]).values : null;
  const name = trace.raw.dataset?.features?.[0] ?? "x";
  return (
    <figure className="nvz-scene">
      <figcaption>
        Predicted probability <span className="mono">σ({fmt(w.values[0])}·{name} + {fmt(b.values[0])})</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Input versus predicted probability">
        <line className="nvz-axis" x1={sx(lo)} y1={sy(0)} x2={sx(hi)} y2={sy(0)} />
        <line className="nvz-axis" x1={sx(0)} y1={sy(0)} x2={sx(0)} y2={sy(1)} />
        <line className="nvz-grid" x1={sx(lo)} y1={sy(0.5)} x2={sx(hi)} y2={sy(0.5)} />
        <text className="nvz-axis-label" x={W - pad} y={sy(0) - 6}>{name}</text>
        <text className="nvz-axis-label" x={sx(0) + 6} y={pad}>p</text>
        <text className="nvz-axis-label" x={sx(lo)} y={sy(0.5) - 4}>0.5</text>
        {prevW && prevB && curve(prevW, prevB, "nvz-curve prev")}
        {curve(w, b, "nvz-curve")}
        {xs.map((px, i) => {
          const w0 = w.values![0];
          const b0 = b.values![0];
          const p = probs ? probs[i] : 1 / (1 + Math.exp(-(w0 * px + b0)));
          const l = label(i);
          return (
            <g key={i} onClick={() => onPoint(i)}>
              {l !== null && <circle className={pointClass(i)} cx={sx(px)} cy={sy(l)} r={7}><title>{title(i)}</title></circle>}
              <circle className={`nvz-pt nvz-pt-pred ${i === sampleIndex ? "current" : ""}`} cx={sx(px)} cy={sy(p)} r={5}>
                <title>{`sample ${i}: p = ${fmt(p)}`}</title>
              </circle>
            </g>
          );
        })}
      </svg>
      <SceneLegend classes={classes} labelled={!!y} prev={!!prevW} pred />
    </figure>
  );
}

function SceneLegend({ classes, labelled, prev, pred }: { classes: string[]; labelled: boolean; prev: boolean; pred?: boolean }) {
  return (
    <ul className="nvz-scene-legend">
      {labelled ? (
        <>
          <li><i className="nvz-pt-0" /> class {classes[0]}</li>
          <li><i className="nvz-pt-1" /> class {classes[1]}</li>
        </>
      ) : (
        <li><i className="nvz-pt-unlabelled" /> sample</li>
      )}
      {pred && <li><i className="nvz-pt-pred" /> predicted p</li>}
      <li><i className="nvz-line" /> current parameters</li>
      {prev && <li><i className="nvz-line prev" /> before this update</li>}
    </ul>
  );
}
