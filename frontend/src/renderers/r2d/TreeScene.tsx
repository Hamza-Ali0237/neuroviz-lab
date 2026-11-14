import { useAppState, useDispatch } from "../../state/store";
import { fmt, ticks } from "../../format";
import { dataBounds, regions, rule, treeAt, type TreeNode, type TreeState } from "../tree";
import { Formula } from "./OpCard";
import { Axes, H, Marker, PAD, SceneLegend, W } from "./LRScene";

const BOX_W = 92;
const BOX_H = 40;

/**
 * x slot of every node: terminals left to right, internal nodes centred over their children.
 * The drawing widens with the number of terminals so boxes never overlap.
 */
function layout(tree: TreeState): { pos: Map<number, { x: number; y: number }>; width: number } {
  const pos = new Map<number, { x: number; y: number }>();
  const byId = new Map(tree.nodes.map((n) => [n.id, n]));
  let slot = 0;
  const depth = Math.max(0, ...tree.nodes.map((n) => n.depth));
  const walk = (id: number): number => {
    const n = byId.get(id)!;
    let x: number;
    if (n.left !== null && n.right !== null && byId.has(n.left) && byId.has(n.right)) x = (walk(n.left) + walk(n.right)) / 2;
    else x = slot++;
    pos.set(id, { x, y: n.depth });
    return x;
  };
  if (byId.has(0)) walk(0);
  const cols = Math.max(1, slot - 1);
  const rows = Math.max(1, depth);
  const width = Math.max(W, slot * (BOX_W + 12) + 2 * PAD);
  for (const p of pos.values()) {
    p.x = PAD + 20 + (p.x / cols) * (width - 2 * PAD - 40);
    p.y = PAD + (p.y / rows) * (H - 2 * PAD - BOX_H / 2);
  }
  return { pos, width };
}

function CountsBar({ counts, x, y, w }: { counts: number[]; x: number; y: number; w: number }) {
  const total = counts.reduce((a, b) => a + b, 0) || 1;
  let acc = 0;
  return (
    <g>
      {counts.map((c, k) => {
        const seg = (c / total) * w;
        const el = <rect key={k} className={`nvz-count-${k}`} x={x + acc} y={y} width={seg} height={5} />;
        acc += seg;
        return el;
      })}
      <title>{`class counts ${counts.join(" / ")}`}</title>
    </g>
  );
}

/**
 * Tree diagram. Split nodes show their rule and class counts, leaves their prediction. In
 * predict traces the current sample's path is highlighted as far as it has been tested; in
 * train traces the node being split and the nodes created in this step are highlighted.
 */
export function TreeDiagram() {
  const { trace, stepIndex, sampleIndex, selection } = useAppState();
  const dispatch = useDispatch();
  const tree = treeAt(trace, stepIndex);
  if (!tree) return <p className="nvz-muted">No tree yet at this step.</p>;
  const { pos, width } = layout(tree);
  const classes = trace.raw.dataset?.classes ?? ["0", "1"];
  const path = tree.paths[sampleIndex] ?? [];
  const onPath = (id: number) => path.includes(id);
  const splitting = tree.split?.node ?? null;
  const x = trace.byRole("input", "x");
  const title = (n: TreeNode) =>
    `node ${n.id} · depth ${n.depth} · ${n.n_samples} samples · counts ${n.counts.join("/")}` +
    (n.impurity !== null ? ` · impurity ${fmt(n.impurity)}` : "") +
    (n.reason ? ` · leaf: ${n.reason}` : "");
  return (
    <figure className="nvz-scene nvz-tree">
      <figcaption>
        Tree{" "}
        <span className="nvz-muted">
          {trace.raw.kind === "predict" ? `path of sample ${sampleIndex} highlighted` : splitting !== null ? `splitting node ${splitting}` : `${tree.nodes.length} nodes so far`}
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${width} ${H}`} role="img" aria-label="Decision tree">
        {tree.nodes.map((n) => {
          const p = pos.get(n.id);
          const parent = n.parent !== null ? pos.get(n.parent) : null;
          if (!p || !parent) return null;
          const lit = onPath(n.id) && onPath(n.parent!);
          return (
            <g key={`e${n.id}`}>
              <line className={`nvz-tree-edge ${lit ? "lit" : ""} ${tree.created.includes(n.id) ? "new" : ""}`} x1={parent.x} y1={parent.y + BOX_H / 2} x2={p.x} y2={p.y - BOX_H / 2} />
              <text className="nvz-tick" x={(parent.x + p.x) / 2 + (n.side === "left" ? -10 : 10)} y={(parent.y + p.y) / 2} textAnchor="middle">
                {n.side === "left" ? "yes" : "no"}
              </text>
            </g>
          );
        })}
        {tree.nodes.map((n) => {
          const p = pos.get(n.id);
          if (!p) return null;
          const terminal = n.is_leaf || n.feature === null;
          const lit = onPath(n.id);
          const current = trace.raw.kind === "predict" ? path[path.length - 1] === n.id : splitting === n.id;
          const cls = `nvz-tree-node ${terminal ? "leaf" : ""} ${lit ? "lit" : ""} ${current ? "current" : ""} ${n.pending ? "pending" : ""} ${tree.created.includes(n.id) ? "new" : ""}`;
          return (
            <g key={n.id} className={cls} transform={`translate(${p.x - BOX_W / 2}, ${p.y - BOX_H / 2})`} onClick={() => x && dispatch({ type: "select", selection: { tensorId: x.id, index: [sampleIndex, 0] } })}>
              <rect width={BOX_W} height={BOX_H} rx={6} />
              <text x={BOX_W / 2} y={15} textAnchor="middle" className="nvz-tree-rule">
                {terminal ? `→ class ${classes[n.predicted_class]} (${fmt(n.probability)})` : rule(n)}
              </text>
              <text x={BOX_W / 2} y={27} textAnchor="middle" className="nvz-tick">
                {n.pending ? "not decided yet" : `n = ${n.n_samples}`}
              </text>
              <CountsBar counts={n.counts} x={8} y={31} w={BOX_W - 16} />
              <title>{title(n)}</title>
            </g>
          );
        })}
      </svg>
      <ul className="nvz-scene-legend">
        <li><i className="nvz-pt-0" /> class {classes[0]} count</li>
        <li><i className="nvz-pt-1" /> class {classes[1]} count</li>
        {trace.raw.kind === "predict" && <li><i className="nvz-tree-lit" /> sample {sampleIndex}'s path</li>}
        {trace.raw.kind === "train" && <li><i className="nvz-tree-lit" /> node being split</li>}
        {selection.tensorId ? null : <li className="nvz-muted">yes = left branch</li>}
      </ul>
    </figure>
  );
}

/** Train: weighted impurity against threshold for every candidate split of the node being split, one line per feature, the chosen one marked. */
export function SplitSearch() {
  const { trace, stepIndex } = useAppState();
  const tree = treeAt(trace, stepIndex);
  const s = tree?.split;
  if (!tree || !s) return null;
  const feats = [...new Set(s.candidates.map((c) => c.feature))];
  const names = tree.features;
  const ts = s.candidates.map((c) => c.threshold);
  const lo = Math.min(...ts);
  const hi = Math.max(...ts);
  const ylo = 0;
  const yhi = Math.max(s.impurity, ...s.candidates.map((c) => c.weighted_impurity)) * 1.05;
  const sx = (v: number) => PAD + ((v - lo) / (hi - lo || 1)) * (W - 2 * PAD);
  const sy = (v: number) => H - PAD - ((v - ylo) / (yhi - ylo || 1)) * (H - 2 * PAD);
  const chosen = s.chosen;
  const label = s.criterion === "gini" ? "G" : "H";
  return (
    <figure className="nvz-scene nvz-split">
      <figcaption>
        Split search at node {s.node} <span className="nvz-muted">({s.n_samples} samples, counts {s.counts.join("/")}, {s.criterion} impurity {fmt(s.impurity)})</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Weighted impurity against threshold">
        <Axes sx={sx} sy={sy} xlo={lo} xhi={hi} ylo={ylo} yhi={yhi} xname="threshold" yname={`weighted ${label}`} yTicks={ticks(ylo, yhi, 4)} />
        <line className="nvz-boundary prev" x1={sx(lo)} y1={sy(s.impurity)} x2={sx(hi)} y2={sy(s.impurity)} />
        <text className="nvz-tick" x={sx(hi)} y={sy(s.impurity) - 4} textAnchor="end">parent {fmt(s.impurity)}</text>
        {feats.map((f) => {
          const pts = s.candidates.filter((c) => c.feature === f).sort((a, b) => a.threshold - b.threshold);
          return (
            <g key={f} className={`nvz-split-line nvz-split-f${f}`}>
              <polyline points={pts.map((c) => `${sx(c.threshold).toFixed(1)},${sy(c.weighted_impurity).toFixed(1)}`).join(" ")} />
              {pts.map((c, k) => (
                <circle key={k} cx={sx(c.threshold)} cy={sy(c.weighted_impurity)} r={2.2}>
                  <title>{`${c.feature_name} ≤ ${fmt(c.threshold)}: weighted ${fmt(c.weighted_impurity)} (left ${c.left_counts.join("/")} ${label}=${fmt(c.left_impurity)}, right ${c.right_counts.join("/")} ${label}=${fmt(c.right_impurity)}), gain ${fmt(c.gain)}`}</title>
                </circle>
              ))}
            </g>
          );
        })}
        {chosen && (
          <g className="nvz-split-chosen">
            <circle cx={sx(chosen.threshold)} cy={sy(chosen.weighted_impurity)} r={8} />
            <text x={sx(chosen.threshold)} y={sy(chosen.weighted_impurity) + 24} textAnchor="middle" className="nvz-axis-label">
              {chosen.feature_name} ≤ {fmt(chosen.threshold)} · gain {fmt(chosen.gain)}
            </text>
          </g>
        )}
      </svg>
      <ul className="nvz-scene-legend">
        {feats.map((f) => (
          <li key={f}><i className={`nvz-split-swatch nvz-split-f${f}`} /> threshold on {names[f] ?? `x${f + 1}`}</li>
        ))}
        <li><i className="nvz-line prev" /> impurity before the split</li>
        <li><i className="nvz-split-ring" /> chosen: largest gain</li>
      </ul>
      <div className="nvz-split-formula">
        <b>{s.explain.title}.</b> <Formula tex={s.explain.formula} />
        <p className="nvz-text">{s.explain.text}</p>
        <p className="nvz-muted">
          Each candidate threshold is the midpoint between two consecutive sorted values of one feature. Its score is the sample-weighted impurity of the two
          children; the chosen split has the largest gain = parent impurity − weighted impurity ({s.candidates.length} candidates tried).
        </p>
      </div>
    </figure>
  );
}

/** Feature-space partition: the dataset with the split lines and leaf regions shaded (2 features) or bands (1 feature), as the tree stands at this step. */
export function Partition() {
  const { trace, stepIndex, sampleIndex, selection } = useAppState();
  const dispatch = useDispatch();
  const tree = treeAt(trace, stepIndex);
  const bounds = dataBounds(trace);
  const x = trace.byRole("input", "x");
  const y = trace.byRole("label");
  if (!tree || !bounds) return null;
  const nf = bounds.length;
  const isTrain = trace.raw.kind === "train";
  // Train traces grow the tree on the whole dataset, so the dataset is the traced x itself.
  const ctxX = trace.context("dataset_X") ?? (isTrain ? x : null);
  const ctxY = trace.context("dataset_y") ?? (isTrain ? y : null);
  const names = trace.raw.dataset?.features ?? ["x1", "x2"];
  const classes = trace.raw.dataset?.classes ?? ["0", "1"];
  const [lo1, hi1] = bounds[0];
  const [lo2, hi2] = nf > 1 ? bounds[1] : [0, 1];
  const sx = (v: number) => PAD + ((v - lo1) / (hi1 - lo1)) * (W - 2 * PAD);
  const sy = (v: number) => H - PAD - ((v - lo2) / (hi2 - lo2)) * (H - 2 * PAD);
  const regs = regions(tree, bounds);
  const cls = (l: number | null | undefined) => (l === null || l === undefined ? null : l > 0.5 ? 1 : 0);
  const traced = isTrain ? 0 : trace.raw.mode.n_samples;
  const selRow = x && selection.tensorId === x.id && selection.index ? selection.index[0] : -1;
  const splitting = tree.split?.node ?? null;
  const path = tree.paths[sampleIndex] ?? [];
  const py = (k: number, l: number | null) => (nf > 1 ? sy(ctxX!.values![k * nf + 1]) : sy(l === null ? 0.5 : 0.2 + 0.6 * l));
  return (
    <figure className="nvz-scene">
      <figcaption>
        Feature space <span className="nvz-muted">{regs.length} region{regs.length === 1 ? "" : "s"}, shaded by the majority class of each node</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Feature-space partition">
        {regs.map(({ node, bounds: b }) => {
          const x0 = sx(b[0][0]);
          const x1 = sx(b[0][1]);
          const y0 = nf > 1 ? sy(b[1][1]) : PAD;
          const y1 = nf > 1 ? sy(b[1][0]) : H - PAD;
          const hot = node.id === splitting || (path.length > 0 && path[path.length - 1] === node.id);
          return (
            <rect key={node.id} className={`nvz-region nvz-region-${node.predicted_class} ${hot ? "hot" : ""} ${node.pending ? "pending" : ""}`} x={x0} y={y0} width={x1 - x0} height={y1 - y0} style={{ opacity: 0.12 + 0.3 * node.probability }}>
              <title>{`node ${node.id}: class ${classes[node.predicted_class]} with p = ${fmt(node.probability)} (counts ${node.counts.join("/")})${node.pending ? " · not decided yet" : ""}`}</title>
            </rect>
          );
        })}
        {tree.nodes.filter((n) => n.feature !== null && n.threshold !== null).map((n) => {
          const b = boundsOf(tree, n.id, bounds);
          if (!b) return null;
          const t = n.threshold!;
          return n.feature === 0 ? (
            <line key={n.id} className={`nvz-split-rule ${n.id === splitting ? "hot" : ""}`} x1={sx(t)} x2={sx(t)} y1={nf > 1 ? sy(b[1][1]) : PAD} y2={nf > 1 ? sy(b[1][0]) : H - PAD} />
          ) : (
            <line key={n.id} className={`nvz-split-rule ${n.id === splitting ? "hot" : ""}`} y1={sy(t)} y2={sy(t)} x1={sx(b[0][0])} x2={sx(b[0][1])} />
          );
        })}
        <Axes sx={sx} sy={sy} xlo={lo1} xhi={hi1} ylo={lo2} yhi={hi2} xname={names[0]} yname={nf > 1 ? names[1] : "class"} yTicks={nf > 1 ? undefined : []} />
        {ctxX?.values &&
          Array.from({ length: ctxX.shape[0] }, (_, k) => {
            const c = cls(ctxY?.values?.[k]);
            return <Marker key={k} cls={c} cx={sx(ctxX.values![k * nf])} cy={py(k, ctxY?.values?.[k] ?? null)} r={isTrain ? 4.5 : 3.5} className={`${isTrain ? "nvz-pt nvz-pt-data" : "nvz-ctx-pt"} ${c === null ? "nvz-pt-unlabelled" : `nvz-pt-${c}`}`} />;
          })}
        {x?.values &&
          Array.from({ length: traced }, (_, i) => {
            const c = cls(y?.values?.[i]);
            const l = y?.values?.[i] ?? null;
            return (
              <Marker
                key={i}
                cls={c}
                cx={sx(x.values![i * nf])}
                cy={nf > 1 ? sy(x.values![i * nf + 1]) : sy(l === null ? 0.5 : 0.2 + 0.6 * l)}
                r={7}
                className={`nvz-pt ${c === null ? "nvz-pt-unlabelled" : `nvz-pt-${c}`} ${i === sampleIndex ? "current" : ""} ${i === selRow ? "selected" : ""}`}
                onClick={() => {
                  dispatch({ type: "sample", index: i });
                  dispatch({ type: "select", selection: { tensorId: x.id, index: [i, 0] } });
                }}
              >
                <title>{`sample ${i}: ${names[0]} = ${fmt(x.values![i * nf])}${nf > 1 ? `, ${names[1]} = ${fmt(x.values![i * nf + 1])}` : ""}`}</title>
              </Marker>
            );
          })}
      </svg>
      <SceneLegend classes={classes} labelled={!!(y || ctxY)} prev={false} context={!isTrain && !!ctxX} line={false} />
      <ul className="nvz-scene-legend">
        <li><i className="nvz-region-swatch nvz-region-0" /> region → class {classes[0]}</li>
        <li><i className="nvz-region-swatch nvz-region-1" /> region → class {classes[1]}</li>
        <li><i className="nvz-line" /> split threshold</li>
      </ul>
      {ctxX?.note && <p className="nvz-ctx-note">{ctxX.note}</p>}
    </figure>
  );
}

/** Bounds of the region a node governs: the root bounds narrowed by every ancestor's split. */
function boundsOf(tree: TreeState, id: number, root: [number, number][]): [number, number][] | null {
  const byId = new Map(tree.nodes.map((n) => [n.id, n]));
  const chain: TreeNode[] = [];
  for (let n = byId.get(id); n; n = n.parent !== null ? byId.get(n.parent) : undefined) chain.unshift(n);
  if (!chain.length || chain[0].id !== 0) return null;
  const b = root.map((r) => [...r] as [number, number]);
  for (let k = 0; k < chain.length - 1; k++) {
    const p = chain[k];
    const child = chain[k + 1];
    if (p.feature === null || p.threshold === null) break;
    if (child.side === "left") b[p.feature][1] = Math.min(b[p.feature][1], p.threshold);
    else b[p.feature][0] = Math.max(b[p.feature][0], p.threshold);
  }
  return b;
}

/**
 * All tree panels. Train traces at a split step lead with the split search (the "how a tree
 * decides" moment). In capture mode one panel fills the stage: `?scene=split|tree|partition`
 * (default: the split search when there is one, else the tree).
 */
export function TreeScene() {
  const { trace, stepIndex, capture, scene } = useAppState();
  const tree = treeAt(trace, stepIndex);
  const hasSplit = !!tree?.split;
  if (capture) {
    const which = scene ?? (hasSplit ? "split" : "tree");
    if (which === "split") return hasSplit ? <SplitSearch /> : <TreeDiagram />;
    if (which === "partition") return <Partition />;
    return <TreeDiagram />;
  }
  return (
    <div className="nvz-tree-scene">
      {hasSplit && <SplitSearch />}
      <TreeDiagram />
      <Partition />
    </div>
  );
}
