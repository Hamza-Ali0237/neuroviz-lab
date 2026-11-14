import { useAppState, useDispatch } from "../../state/store";
import { fmt, signed, ticks } from "../../format";
import type { Tensor } from "../../trace/load";

export const W = 420;
export const H = 300;
export const PAD = 40;

/** Class marker: circle for class 0, diamond for class 1, so classes differ by shape too. */
export function Marker({ cls, cx, cy, r, className, onClick, children }: {
  cls: number | null;
  cx: number;
  cy: number;
  r: number;
  className: string;
  onClick?: () => void;
  children?: React.ReactNode;
}) {
  if (cls === 1) {
    const d = `M${cx},${cy - r * 1.25} L${cx + r * 1.25},${cy} L${cx},${cy + r * 1.25} L${cx - r * 1.25},${cy} Z`;
    return (
      <path className={className} d={d} onClick={onClick}>
        {children}
      </path>
    );
  }
  return (
    <circle className={className} cx={cx} cy={cy} r={r} onClick={onClick}>
      {children}
    </circle>
  );
}

export function Axes({ sx, sy, xlo, xhi, ylo, yhi, xname, yname, yTicks }: {
  sx: (v: number) => number;
  sy: (v: number) => number;
  xlo: number;
  xhi: number;
  ylo: number;
  yhi: number;
  xname: string;
  yname: string;
  yTicks?: number[];
}) {
  const xt = ticks(xlo, xhi);
  const yt = yTicks ?? ticks(ylo, yhi);
  return (
    <g className="nvz-axes">
      {xt.map((v) => (
        <g key={`x${v}`}>
          <line className="nvz-grid" x1={sx(v)} y1={sy(ylo)} x2={sx(v)} y2={sy(yhi)} />
          <text className="nvz-tick" x={sx(v)} y={H - PAD + 14} textAnchor="middle">{fmt(v)}</text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={`y${v}`}>
          <line className="nvz-grid" x1={sx(xlo)} y1={sy(v)} x2={sx(xhi)} y2={sy(v)} />
          <text className="nvz-tick" x={PAD - 6} y={sy(v) + 4} textAnchor="end">{fmt(v)}</text>
        </g>
      ))}
      <line className="nvz-axis" x1={sx(xlo)} y1={sy(ylo)} x2={sx(xhi)} y2={sy(ylo)} />
      <line className="nvz-axis" x1={sx(xlo)} y1={sy(ylo)} x2={sx(xlo)} y2={sy(yhi)} />
      <text className="nvz-axis-label" x={W - PAD} y={H - 6} textAnchor="end">{xname}</text>
      <text className="nvz-axis-label" x={PAD + 4} y={PAD - 8}>{yname}</text>
    </g>
  );
}

/**
 * Logistic-regression scene. 2 features: scatter of the samples with the decision
 * boundary w·x + b = 0 from the parameters current at this step. 1 feature: x vs
 * predicted probability with the sigmoid curve σ(w x + b). The whole dataset (context
 * tensors) is drawn faded behind the traced samples. At an update step the previous
 * boundary/curve is dashed so the move is visible.
 */
export function LRScene() {
  const { trace, stepIndex, sampleIndex, selection } = useAppState();
  const dispatch = useDispatch();
  const x = trace.byRole("input", "x");
  const y = trace.byRole("label");
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
  const ctxX = trace.context("dataset_X");
  const ctxY = trace.context("dataset_y");
  const ctxN = ctxX?.values && ctxX.shape[1] === nf ? ctxX.shape[0] : 0;
  const label = (i: number) => (y?.values ? y.values[i] : null);
  const cls = (l: number | null) => (l === null ? null : l > 0.5 ? 1 : 0);
  const selRow = selection.index && selection.tensorId !== w.id && selection.tensorId !== b.id ? selection.index[0] : -1;
  const classes = trace.raw.dataset?.classes ?? ["0", "1"];
  const pointClass = (i: number) => {
    const c = cls(label(i));
    const k = c === null ? "nvz-pt-unlabelled" : `nvz-pt-${c}`;
    return `nvz-pt ${k} ${i === sampleIndex ? "current" : ""} ${i === selRow ? "selected" : ""}`;
  };
  const onPoint = (i: number) => {
    dispatch({ type: "sample", index: i });
    dispatch({ type: "select", selection: { tensorId: x.id, index: [i, 0] } });
  };
  const title = (i: number) => {
    const l = label(i);
    const coords = Array.from({ length: nf }, (_, j) => fmt(x.values![i * nf + j])).join(", ");
    return `sample ${i}: x = (${coords})${l === null ? "" : `, y = ${fmt(l)}`}`;
  };
  const contextPoints = (px: (k: number) => number, py: (k: number) => number) =>
    Array.from({ length: ctxN }, (_, k) => {
      const c = ctxY?.values ? cls(ctxY.values[k]) : null;
      return <Marker key={k} cls={c} cx={px(k)} cy={py(k)} r={3.5} className={`nvz-ctx-pt ${c === null ? "nvz-pt-unlabelled" : `nvz-pt-${c}`}`} />;
    });
  const names = trace.raw.dataset?.features ?? (nf === 2 ? ["x1", "x2"] : ["x"]);

  if (nf === 2) {
    const xs = Array.from({ length: n }, (_, i) => x.values![i * 2]);
    const ys = Array.from({ length: n }, (_, i) => x.values![i * 2 + 1]);
    const all = [...xs, ...ys, ...(ctxX?.values ?? [])];
    const lo = Math.floor(Math.min(-2.5, ...all) - 0.5);
    const hi = Math.ceil(Math.max(2.5, ...all) + 0.5);
    const sx = (v: number) => PAD + ((v - lo) / (hi - lo)) * (W - 2 * PAD);
    const sy = (v: number) => H - PAD - ((v - lo) / (hi - lo)) * (H - 2 * PAD);
    const boundary = (wt: Tensor, bt: Tensor, kls: string) => {
      const [w1, w2] = wt.values!;
      const b0 = bt.values![0];
      // w1·x1 + w2·x2 + b = 0
      if (Math.abs(w2) < 1e-9 && Math.abs(w1) < 1e-9) return null;
      const pts =
        Math.abs(w2) >= Math.abs(w1)
          ? [lo, hi].map((x1) => [x1, -(w1 * x1 + b0) / w2])
          : [lo, hi].map((x2) => [-(w2 * x2 + b0) / w1, x2]);
      return <line className={kls} x1={sx(pts[0][0])} y1={sy(pts[0][1])} x2={sx(pts[1][0])} y2={sy(pts[1][1])} />;
    };
    return (
      <figure className="nvz-scene">
        <figcaption>
          Decision boundary{" "}
          <span className="mono">
            {fmt(w.values[0])}·{names[0]}{signed(w.values[1])}·{names[1]}{signed(b.values[0])} = 0
          </span>
        </figcaption>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Samples and decision boundary">
          <defs>
            <clipPath id="nvz-plot-clip"><rect x={PAD} y={PAD} width={W - 2 * PAD} height={H - 2 * PAD} /></clipPath>
          </defs>
          <Axes sx={sx} sy={sy} xlo={lo} xhi={hi} ylo={lo} yhi={hi} xname={names[0]} yname={names[1]} />
          <g clipPath="url(#nvz-plot-clip)">
            {contextPoints((k) => sx(ctxX!.values![k * 2]), (k) => sy(ctxX!.values![k * 2 + 1]))}
            {prevW && prevB && boundary(prevW, prevB, "nvz-boundary prev")}
            {boundary(w, b, "nvz-boundary")}
          </g>
          {xs.map((px, i) => (
            <Marker key={i} cls={cls(label(i))} cx={sx(px)} cy={sy(ys[i])} r={7} className={pointClass(i)} onClick={() => onPoint(i)}>
              <title>{title(i)}</title>
            </Marker>
          ))}
        </svg>
        <SceneLegend classes={classes} labelled={!!y} prev={!!prevW} context={ctxN > 0} />
        {ctxX?.note && ctxN > 0 && <p className="nvz-ctx-note">{ctxX.note}</p>}
      </figure>
    );
  }

  // 1 feature: x vs probability with the sigmoid curve.
  const xs = Array.from({ length: n }, (_, i) => x.values![i]);
  const lo = Math.floor(Math.min(-4, ...xs, ...(ctxX?.values ?? [])) - 0.5);
  const hi = Math.ceil(Math.max(4, ...xs, ...(ctxX?.values ?? [])) + 0.5);
  const sx = (v: number) => PAD + ((v - lo) / (hi - lo)) * (W - 2 * PAD);
  const sy = (p: number) => H - PAD - p * (H - 2 * PAD);
  const sig = (wt: Tensor, bt: Tensor, xv: number) => 1 / (1 + Math.exp(-(wt.values![0] * xv + bt.values![0])));
  const curve = (wt: Tensor, bt: Tensor, kls: string) => {
    const pts: string[] = [];
    for (let k = 0; k <= 80; k++) {
      const xv = lo + ((hi - lo) * k) / 80;
      pts.push(`${sx(xv).toFixed(1)},${sy(sig(wt, bt, xv)).toFixed(1)}`);
    }
    return <polyline className={kls} points={pts.join(" ")} />;
  };
  const out = trace.nodesInStep(stepIndex).find((nd) => nd.op === "sigmoid" && nd.phase === "forward");
  const probs = out ? trace.tensor(out.outputs[0]).values : null;
  return (
    <figure className="nvz-scene">
      <figcaption>
        Predicted probability <span className="mono">σ({fmt(w.values[0])}·{names[0]}{signed(b.values[0])})</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Input versus predicted probability">
        <Axes sx={sx} sy={sy} xlo={lo} xhi={hi} ylo={0} yhi={1} xname={names[0]} yname="p" yTicks={[0, 0.25, 0.5, 0.75, 1]} />
        {contextPoints((k) => sx(ctxX!.values![k]), (k) => sy(ctxY?.values ? ctxY.values[k] : 0.5))}
        {prevW && prevB && curve(prevW, prevB, "nvz-curve prev")}
        {curve(w, b, "nvz-curve")}
        {xs.map((px, i) => {
          const p = probs ? probs[i] : sig(w, b, px);
          const l = label(i);
          return (
            <g key={i} onClick={() => onPoint(i)}>
              {l !== null && (
                <Marker cls={cls(l)} cx={sx(px)} cy={sy(l)} r={7} className={pointClass(i)}>
                  <title>{title(i)}</title>
                </Marker>
              )}
              <circle className={`nvz-pt nvz-pt-pred ${i === sampleIndex ? "current" : ""}`} cx={sx(px)} cy={sy(p)} r={5}>
                <title>{`sample ${i}: p = ${fmt(p)}`}</title>
              </circle>
            </g>
          );
        })}
      </svg>
      <SceneLegend classes={classes} labelled={!!y} prev={!!prevW} pred context={ctxN > 0} />
      {ctxX?.note && ctxN > 0 && <p className="nvz-ctx-note">{ctxX.note}</p>}
    </figure>
  );
}

export function SceneLegend({ classes, labelled, prev, pred, context, line = true }: {
  classes: string[];
  labelled: boolean;
  prev: boolean;
  pred?: boolean;
  context?: boolean;
  line?: boolean;
}) {
  return (
    <ul className="nvz-scene-legend">
      {labelled ? (
        <>
          <li><i className="nvz-pt-0" /> class {classes[0]} (circle)</li>
          <li><i className="nvz-pt-1 diamond" /> class {classes[1]} (diamond)</li>
        </>
      ) : (
        <li><i className="nvz-pt-unlabelled" /> sample</li>
      )}
      {pred && <li><i className="nvz-pt-pred" /> predicted p</li>}
      {context && <li><i className="nvz-pt-ctx" /> whole dataset (faded)</li>}
      {line && <li><i className="nvz-line" /> current parameters</li>}
      {prev && <li><i className="nvz-line prev" /> before this update</li>}
    </ul>
  );
}
