import { useAppState, useDispatch } from "../../state/store";
import { fmt, NEG_RGB, POS_RGB, tint } from "../../format";
import { mlpView, probGridAt, type EdgeView } from "../mlp";
import { Axes, Marker, H, PAD, SceneLegend, W } from "./LRScene";

const rgb = (c: [number, number, number]) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
/** Fits inside a neuron: 3 significant digits, the full value stays in the hover title. */
const fmtNeuron = (v: number) => (v === 0 ? "0" : Number(v.toPrecision(3)).toString().replace("-", "−"));

/**
 * Network diagram (2 → 4 → 2). Neurons: circles filled by the activation for the selected
 * sample on the diverging scale (per-layer |max|). Edges: weight sign as colour (blue +,
 * orange −) and |weight| as thickness; on backward steps the gradient of the weight instead.
 * Layers light up as their step is reached. Click a neuron to select its value.
 */
export function NetworkDiagram() {
  const { trace, stepIndex, sampleIndex, selection } = useAppState();
  const dispatch = useDispatch();
  const view = mlpView(trace, stepIndex, sampleIndex);
  if (!view) return null;
  const { layers, edges, backward } = view;
  const cols = layers.map((_, l) => PAD + 30 + (l * (W - 2 * PAD - 60)) / (layers.length - 1));
  const ys = (size: number) => Array.from({ length: size }, (_, j) => H / 2 + (j - (size - 1) / 2) * 52);
  const R = 19;
  const edgeLines = (e: EdgeView, l: number) => {
    const t = e.tensor;
    const [nIn, nOut] = t.shape;
    const absMax = Math.max(Math.abs(t.stats.min), Math.abs(t.stats.max), 1e-12);
    const yi = ys(nIn);
    const yo = ys(nOut);
    const out: JSX.Element[] = [];
    for (let i = 0; i < nIn; i++) {
      for (let j = 0; j < nOut; j++) {
        const v = t.values![i * nOut + j];
        const label = `${e.grad ? "∂L/∂" : ""}${e.name}[${i}, ${j}] = ${v}`;
        const sel = selection.tensorId === t.id && selection.index?.[0] === i && selection.index?.[1] === j;
        out.push(
          <line
            key={`${l}-${i}-${j}`}
            className={`nvz-edge ${sel ? "selected" : ""}`}
            x1={cols[l] + R} y1={yi[i]} x2={cols[l + 1] - R} y2={yo[j]}
            stroke={rgb(v >= 0 ? POS_RGB : NEG_RGB)}
            strokeWidth={0.6 + 5 * (Math.abs(v) / absMax)}
            strokeOpacity={0.35 + 0.65 * (Math.abs(v) / absMax)}
            onClick={() => dispatch({ type: "select", selection: { tensorId: t.id, index: [i, j] } })}
          >
            <title>{label}</title>
          </line>,
        );
      }
    }
    return out;
  };
  const what = backward ? (edges.every((e) => e.grad) ? "gradients ∂L/∂W" : "gradients where computed so far, weights elsewhere") : "weights";
  return (
    <figure className="nvz-scene nvz-network">
      <figcaption>
        Network for sample {sampleIndex}: <span className="nvz-muted">neurons = activations, edges = {what}</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Network diagram">
        {edges.map((e, l) => edgeLines(e, l))}
        {layers.map((layer, l) =>
          ys(layer.size).map((y, j) => {
            const v = layer.values?.[j] ?? null;
            const lit = v !== null;
            const sel = selection.tensorId === layer.tensorId && selection.index?.[0] === sampleIndex && selection.index?.[1] === j;
            return (
              <g
                key={`${l}-${j}`}
                className={`nvz-neuron ${lit ? "lit" : ""} ${sel ? "selected" : ""}`}
                onClick={() => layer.tensorId && dispatch({ type: "select", selection: { tensorId: layer.tensorId, index: [sampleIndex, j] } })}
              >
                <circle cx={cols[l]} cy={y} r={R} style={{ fill: lit ? tint(v!, layer.absMax) : "var(--panel)" }} />
                <text x={cols[l]} y={y + 4} textAnchor="middle" className="nvz-neuron-value">
                  {lit ? fmtNeuron(v!) : "·"}
                </text>
                <title>{lit && layer.tensorId ? `${trace.label(layer.tensorId)}[${sampleIndex}, ${j}] = ${v}` : `${layer.label}: not computed yet at this step`}</title>
              </g>
            );
          }),
        )}
        {layers.map((layer, l) => (
          <text key={l} x={cols[l]} y={H - 10} textAnchor="middle" className="nvz-axis-label">
            {layer.label}
          </text>
        ))}
        {edges.map((e, l) => (
          <text key={l} x={(cols[l] + cols[l + 1]) / 2} y={PAD - 12} textAnchor="middle" className="nvz-axis-label">
            {e.grad ? `∂L/∂${e.name}` : e.name} ({e.tensor.shape.join("×")})
          </text>
        ))}
      </svg>
      <ul className="nvz-scene-legend">
        <li><i className="nvz-edge-pos" /> positive</li>
        <li><i className="nvz-edge-neg" /> negative</li>
        <li>thickness ∝ |{backward ? "gradient" : "weight"}|</li>
        <li><i className="nvz-neuron-dim" /> not computed yet</li>
      </ul>
    </figure>
  );
}

/** Decision regions over moons from the probability grid current at this step; the dataset faded, traced samples highlighted. */
export function DecisionRegions() {
  const { trace, stepIndex, sampleIndex, selection } = useAppState();
  const dispatch = useDispatch();
  const view = mlpView(trace, stepIndex, sampleIndex);
  const g = probGridAt(trace, view?.reached ?? 0);
  const x = trace.byRole("input", "x");
  if (!g || !x?.values) return null;
  const { grid, ax1, ax2 } = g;
  const y = trace.byRole("label");
  const ctxX = trace.context("dataset_X");
  const ctxY = trace.context("dataset_y");
  const n1 = ax1.length;
  const n2 = ax2.length;
  const [lo1, hi1, lo2, hi2] = [ax1[0], ax1[n1 - 1], ax2[0], ax2[n2 - 1]];
  const sx = (v: number) => PAD + ((v - lo1) / (hi1 - lo1)) * (W - 2 * PAD);
  const sy = (v: number) => H - PAD - ((v - lo2) / (hi2 - lo2)) * (H - 2 * PAD);
  const cw = (W - 2 * PAD) / (n1 - 1);
  const ch = (H - 2 * PAD) / (n2 - 1);
  const names = trace.raw.dataset?.features ?? ["x1", "x2"];
  const classes = trace.raw.dataset?.classes ?? ["0", "1"];
  const cls = (l: number | null | undefined) => (l === null || l === undefined ? null : l > 0.5 ? 1 : 0);
  const n = trace.raw.mode.n_samples;
  const selRow = selection.tensorId === x.id && selection.index ? selection.index[0] : -1;
  const cells: JSX.Element[] = [];
  for (let i = 0; i < n1; i++) {
    for (let j = 0; j < n2; j++) {
      const p = grid.values![i * n2 + j];
      cells.push(
        <rect
          key={`${i}-${j}`}
          x={sx(ax1[i]) - cw / 2}
          y={sy(ax2[j]) - ch / 2}
          width={cw + 0.5}
          height={ch + 0.5}
          style={{ fill: `color-mix(in srgb, var(--pt1) ${Math.round(p * 100)}%, var(--pt0))` }}
          opacity={0.45}
        >
          <title>{`${grid.name}[${i}, ${j}] = ${p}  (x1 = ${fmt(ax1[i])}, x2 = ${fmt(ax2[j])})`}</title>
        </rect>,
      );
    }
  }
  return (
    <figure className="nvz-scene">
      <figcaption>
        Decision regions <span className="nvz-muted">p(class {classes[1]}) from {grid.name} ({view?.reached ? `after update ${view.reached}` : trace.raw.kind === "train" ? "before training" : "pretrained"})</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Decision regions">
        <defs>
          <clipPath id="nvz-regions-clip"><rect x={PAD} y={PAD} width={W - 2 * PAD} height={H - 2 * PAD} /></clipPath>
        </defs>
        <g clipPath="url(#nvz-regions-clip)">{cells}</g>
        <Axes sx={sx} sy={sy} xlo={lo1} xhi={hi1} ylo={lo2} yhi={hi2} xname={names[0]} yname={names[1]} />
        {ctxX?.values && ctxX.shape[1] === 2 &&
          Array.from({ length: ctxX.shape[0] }, (_, k) => {
            const c = cls(ctxY?.values?.[k]);
            return <Marker key={k} cls={c} cx={sx(ctxX.values![k * 2])} cy={sy(ctxX.values![k * 2 + 1])} r={3.5} className={`nvz-ctx-pt ${c === null ? "nvz-pt-unlabelled" : `nvz-pt-${c}`}`} />;
          })}
        {Array.from({ length: n }, (_, i) => {
          const c = cls(y?.values?.[i]);
          return (
            <Marker
              key={i}
              cls={c}
              cx={sx(x.values![i * 2])}
              cy={sy(x.values![i * 2 + 1])}
              r={7}
              className={`nvz-pt ${c === null ? "nvz-pt-unlabelled" : `nvz-pt-${c}`} ${i === sampleIndex ? "current" : ""} ${i === selRow ? "selected" : ""}`}
              onClick={() => {
                dispatch({ type: "sample", index: i });
                dispatch({ type: "select", selection: { tensorId: x.id, index: [i, 0] } });
              }}
            >
              <title>{`sample ${i}: x = (${fmt(x.values![i * 2])}, ${fmt(x.values![i * 2 + 1])})`}</title>
            </Marker>
          );
        })}
      </svg>
      <SceneLegend classes={classes} labelled={!!y} prev={false} context={!!ctxX} line={false} />
      <p className="nvz-ctx-note">{grid.note}</p>
    </figure>
  );
}

/** Both MLP panels side by side (the network first). In capture mode one panel fills the stage: the network, or the regions with `?scene=regions`. */
export function MLPScene() {
  const { capture, scene } = useAppState();
  if (capture) return scene === "regions" ? <DecisionRegions /> : <NetworkDiagram />;
  return (
    <div className="nvz-mlp-scene">
      <NetworkDiagram />
      <DecisionRegions />
    </div>
  );
}
