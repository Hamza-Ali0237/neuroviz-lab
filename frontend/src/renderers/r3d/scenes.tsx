import { useMemo } from "react";
import * as THREE from "three";
import { Line } from "@react-three/drei";
import { useAppState, useDispatch } from "../../state/store";
import { fmt } from "../../format";
import { unflatIndex } from "../../trace/load";
import { Axes, Label, SIZE, Surface, bilinear, cellColor, span, useTheme } from "./common";

export type SceneId = "probability" | "loss" | "tensors";

export interface SceneDef {
  id: SceneId;
  title: string;
  render: () => JSX.Element;
}

const GRID = 41;

/** LR with 2 features: p(x1, x2) from the parameters current at this step. Height = probability. */
export function ProbabilitySurface() {
  const { trace, stepIndex, sampleIndex, selection } = useAppState();
  const dispatch = useDispatch();
  const theme = useTheme();
  const x = trace.byRole("input", "x")!;
  const y = trace.byRole("label");
  const w = trace.paramsAt(stepIndex).get("w")!;
  const b = trace.paramsAt(stepIndex).get("b")!;
  const ctxX = trace.context("dataset_X");
  const ctxY = trace.context("dataset_y");
  const n = trace.raw.mode.n_samples;
  const names = trace.raw.dataset?.features ?? ["x1", "x2"];
  const all = [...x.values!, ...(ctxX?.values ?? [])];
  const lo = Math.floor(Math.min(-2.5, ...all) - 0.5);
  const hi = Math.ceil(Math.max(2.5, ...all) + 0.5);
  const sx = span(lo, hi);
  const sy = (p: number) => p * (SIZE / 2);
  const [w1, w2] = w.values!;
  const b0 = b.values![0];
  const p = (x1: number, x2: number) => 1 / (1 + Math.exp(-(w1 * x1 + w2 * x2 + b0)));
  const axis = useMemo(() => Array.from({ length: GRID }, (_, k) => lo + ((hi - lo) * k) / (GRID - 1)), [lo, hi]);
  const scene = useMemo(() => axis.map(sx), [axis, sx]);
  const heights = useMemo(() => (i: number, j: number) => sy(p(axis[i], axis[j])), [axis, w1, w2, b0]); // eslint-disable-line react-hooks/exhaustive-deps
  const color = useMemo(
    () => (i: number, j: number) => theme.pt0.clone().lerp(theme.pt1, p(axis[i], axis[j])).lerp(theme.panel, 0.35),
    [axis, w1, w2, b0, theme], // eslint-disable-line react-hooks/exhaustive-deps
  );
  // Decision line w1·x1 + w2·x2 + b = 0 at height p = 0.5, clipped to the data range.
  const line = useMemo(() => {
    const pts: [number, number][] = [];
    const push = (x1: number, x2: number) => x1 >= lo && x1 <= hi && x2 >= lo && x2 <= hi && pts.push([x1, x2]);
    if (Math.abs(w2) > 1e-9) for (const x1 of [lo, hi]) push(x1, -(w1 * x1 + b0) / w2);
    if (Math.abs(w1) > 1e-9) for (const x2 of [lo, hi]) push(-(w2 * x2 + b0) / w1, x2);
    return pts.slice(0, 2);
  }, [w1, w2, b0, lo, hi]);
  const selRow = selection.tensorId === x.id && selection.index ? selection.index[0] : -1;
  const cls = (l: number | null | undefined) => (l === null || l === undefined ? theme.muted : l > 0.5 ? theme.pt1 : theme.pt0);
  return (
    <group position={[0, -SIZE / 4, 0]}>
      <Axes x={{ name: names[0], lo, hi }} z={{ name: names[1], lo, hi }} y={{ name: "p", lo: 0, hi: 1 }} colors={theme} />
      <Surface xs={scene} zs={scene} heights={heights} color={color} opacity={0.85} />
      {line.length === 2 && (
        <Line points={line.map(([x1, x2]) => [sx(x1), sy(0.5) + 0.01, sx(x2)] as [number, number, number])} color={theme.fg} lineWidth={2} />
      )}
      {ctxX?.values &&
        ctxX.shape[1] === 2 &&
        Array.from({ length: ctxX.shape[0] }, (_, k) => (
          <mesh key={k} position={[sx(ctxX.values![k * 2]), 0.03, sx(ctxX.values![k * 2 + 1])]}>
            <sphereGeometry args={[0.035, 8, 8]} />
            <meshStandardMaterial color={cls(ctxY?.values?.[k])} transparent opacity={0.35} />
          </mesh>
        ))}
      {Array.from({ length: n }, (_, i) => {
        const x1 = x.values![i * 2];
        const x2 = x.values![i * 2 + 1];
        const pi = p(x1, x2);
        const current = i === sampleIndex;
        return (
          <group key={i}>
            <Line points={[[sx(x1), 0, sx(x2)], [sx(x1), sy(pi), sx(x2)]]} color={theme.muted} lineWidth={1} dashed dashSize={0.08} gapSize={0.06} />
            <mesh
              position={[sx(x1), sy(pi), sx(x2)]}
              onClick={(e) => {
                e.stopPropagation();
                dispatch({ type: "sample", index: i });
                dispatch({ type: "select", selection: { tensorId: x.id, index: [i, 0] } });
              }}
            >
              <sphereGeometry args={[current ? 0.11 : 0.08, 16, 16]} />
              <meshStandardMaterial color={cls(y?.values?.[i])} emissive={i === selRow ? theme.sel : current ? theme.fg : new THREE.Color(0)} emissiveIntensity={0.5} />
            </mesh>
            {current && <Label at={[sx(x1), sy(pi) + 0.25, sx(x2)]}>{`sample ${i}: p = ${fmt(pi)}`}</Label>}
          </group>
        );
      })}
    </group>
  );
}

/** LR train: the recorded loss grid as a surface; marker at the current params, path across iterations. */
export function LossLandscape() {
  const { trace, stepIndex } = useAppState();
  const theme = useTheme();
  const nf = trace.raw.mode.n_features;
  const grid = trace.context("loss_landscape")!;
  const [n0, n1] = nf === 2 ? ["w1", "w2"] : ["w", "b"];
  const ax0 = trace.context(`loss_landscape_${n0}`)!.values!;
  const ax1 = trace.context(`loss_landscape_${n1}`)!.values!;
  const lo0 = ax0[0];
  const hi0 = ax0[ax0.length - 1];
  const lo1 = ax1[0];
  const hi1 = ax1[ax1.length - 1];
  const s0 = span(lo0, hi0);
  const s1 = span(lo1, hi1);
  const { min, max } = grid.stats;
  const sy = (l: number) => ((l - min) / (max - min || 1)) * (SIZE / 2);
  const xs = useMemo(() => Array.from(ax0, s0), [ax0, s0]);
  const zs = useMemo(() => Array.from(ax1, s1), [ax1, s1]);
  const heights = useMemo(() => (i: number, j: number) => sy(grid.values![i * ax1.length + j]), [grid, ax1, min, max]); // eslint-disable-line react-hooks/exhaustive-deps
  const color = useMemo(
    () => (i: number, j: number) => theme.pt0.clone().lerp(theme.pt1, (grid.values![i * ax1.length + j] - min) / (max - min || 1)).lerp(theme.panel, 0.3),
    [grid, ax1, min, max, theme],
  );
  // Path: the params before the first update, then the `after` of every update of w (and b).
  const point = (params: Map<string, { values: Float32Array | null }>): [number, number] => {
    const w = params.get("w")!.values!;
    const b = params.get("b")!.values!;
    return nf === 2 ? [w[0], w[1]] : [w[0], b[0]];
  };
  const updates = trace.raw.events.filter((e) => e.kind === "sgd_update" && e.payload.param === "w");
  const path = [point(trace.paramsAt(-1)), ...updates.map((e) => point(trace.paramsAt(e.step)))];
  const current = point(trace.paramsAt(stepIndex));
  const toScene = ([a, b]: [number, number]): [number, number, number] => {
    const l = bilinear(grid.values!, ax0, ax1, a, b);
    return [s0(a), l === null ? 0 : sy(l) + 0.02, s1(b)];
  };
  const reached = updates.filter((e) => e.step <= stepIndex).length;
  return (
    <group position={[0, -SIZE / 4, 0]}>
      <Axes x={{ name: n0, lo: lo0, hi: hi0 }} z={{ name: n1, lo: lo1, hi: hi1 }} y={{ name: "loss", lo: min, hi: max }} colors={theme} />
      <Surface xs={xs} zs={zs} heights={heights} color={color} opacity={0.9} />
      {path.length > 1 && <Line points={path.map(toScene)} color={theme.muted} lineWidth={1.5} dashed dashSize={0.1} gapSize={0.06} />}
      {reached > 0 && <Line points={path.slice(0, reached + 1).map(toScene)} color={theme.fg} lineWidth={3} />}
      {path.length > 0 && <Label at={[toScene(path[0])[0], toScene(path[0])[1] - 0.2, toScene(path[0])[2]]} muted>start</Label>}
      {path.map((pt, k) => (
        <mesh key={k} position={toScene(pt)}>
          <sphereGeometry args={[0.045, 12, 12]} />
          <meshStandardMaterial color={k <= reached ? theme.fg : theme.muted} />
        </mesh>
      ))}
      <mesh position={toScene(current)}>
        <sphereGeometry args={[0.09, 16, 16]} />
        <meshStandardMaterial color={theme.accent} emissive={theme.accent} emissiveIntensity={0.4} />
      </mesh>
      <Label at={[toScene(current)[0], toScene(current)[1] + 0.35, toScene(current)[2]]}>
        {`${n0} = ${fmt(current[0])}, ${n1} = ${fmt(current[1])}`}
      </Label>
    </group>
  );
}

/**
 * Generic view for any trace: each forward step's output tensor as a grid of coloured cells,
 * stacked along the depth axis in step order. The current step's tensors are highlighted.
 */
export function TensorStack() {
  const { trace, stepIndex, selection, sampleIndex } = useAppState();
  const dispatch = useDispatch();
  const theme = useTheme();
  const outs = trace.forwardOutputs();
  // On a loss, backward or update step, append that step's own tensors (gradients, updated
  // params) so the current step always has something to select.
  const step = trace.steps[stepIndex];
  if (step.phase !== "forward") {
    const seen = new Set(outs.map((o) => o.tensor.id));
    const add = (tid: string) => !seen.has(tid) && seen.add(tid) && outs.push({ step: step.index, tensor: trace.tensor(tid) });
    for (const n of trace.nodesInStep(stepIndex)) n.outputs.forEach(add);
    for (const e of trace.eventsInStep(stepIndex)) if (e.kind === "sgd_update") add(String(e.payload.after));
  }
  const dims = (t: { shape: number[]; size: number }) => {
    const cols = t.shape.length ? t.shape[t.shape.length - 1] : 1;
    return [t.size / cols, cols] as const;
  };
  const maxDim = Math.max(1, ...outs.map(({ tensor }) => Math.max(...dims(tensor))));
  const cell = Math.min(0.6, (SIZE * 0.9) / maxDim);
  const gap = cell * 0.2;
  const depth = outs.length > 1 ? Math.min(1.4, SIZE / (outs.length - 1)) : 0;
  const z0 = (-(outs.length - 1) * depth) / 2;
  // ponytail: one mesh per cell; switch to instancedMesh if a tensor exceeds a few thousand cells.
  return (
    <group position={[0, -SIZE / 4, 0]}>
      <gridHelper args={[SIZE, 8, theme.border, theme.border]} />
      {outs.map(({ step, tensor: t }, k) => {
        const [rows, cols] = dims(t);
        const absMax = Math.max(Math.abs(t.stats.min), Math.abs(t.stats.max), 1e-12);
        const active = step === stepIndex;
        const z = z0 + k * depth;
        const w = cols * (cell + gap);
        const h = rows * (cell + gap);
        const perSample = t.shape.length >= 1 && t.shape[0] === trace.raw.mode.n_samples && trace.raw.mode.n_samples > 1;
        return (
          <group key={t.id} position={[0, 0.2, z]}>
            <Label at={[0, h + 0.3, 0]} muted={!active}>
              {`${trace.steps[step].title} · ${trace.label(t.id)} (${t.shape.join(", ")})`}
            </Label>
            {t.values
              ? Array.from(t.values, (v, flat) => {
                  const idx = unflatIndex(t.shape, flat);
                  const r = Math.floor(flat / cols);
                  const c = flat % cols;
                  const sel = selection.tensorId === t.id && selection.index?.join(",") === idx.join(",");
                  const sample = perSample && idx[0] === sampleIndex;
                  const select = () => {
                    dispatch({ type: "select", selection: { tensorId: t.id, index: idx } });
                    if (perSample) dispatch({ type: "sample", index: idx[0] });
                  };
                  return (
                    <mesh
                      key={flat}
                      position={[(c - (cols - 1) / 2) * (cell + gap), h - (r + 0.5) * (cell + gap), 0]}
                      onClick={(e) => {
                        e.stopPropagation();
                        select();
                      }}
                    >
                      <boxGeometry args={[cell, cell, sel ? cell * 0.8 : cell * 0.3]} />
                      <meshStandardMaterial
                        color={cellColor(v, absMax, theme.panel)}
                        transparent
                        opacity={active ? 1 : 0.3}
                        emissive={sel ? theme.sel : sample && active ? theme.accent : new THREE.Color(0)}
                        emissiveIntensity={sel ? 0.7 : 0.25}
                      />
                      {active && (
                        <Label at={[0, 0, cell * 0.5]} onClick={select} title={`${trace.label(t.id)}[${idx.join(", ")}] = ${v}`}>
                          <span className="nvz-3d-value">{fmt(v)}</span>
                        </Label>
                      )}
                    </mesh>
                  );
                })
              : null}
            <Line points={[[-w / 2, 0, 0], [w / 2, 0, 0], [w / 2, h, 0], [-w / 2, h, 0], [-w / 2, 0, 0]]} color={active ? theme.fg : theme.border} lineWidth={active ? 2 : 1} />
          </group>
        );
      })}
    </group>
  );
}

/** Scenes available for a trace, in preference order for the current step. */
export function availableScenes(trace: ReturnType<typeof useAppState>["trace"], stepIndex: number): SceneDef[] {
  const out: SceneDef[] = [];
  const linear = trace.raw.model.family === "linear";
  const nf = trace.raw.mode.n_features;
  const hasLoss = !!trace.context("loss_landscape");
  const phase = trace.steps[stepIndex]?.phase;
  const loss: SceneDef = { id: "loss", title: "Loss landscape", render: () => <LossLandscape /> };
  const prob: SceneDef = { id: "probability", title: "Probability surface", render: () => <ProbabilitySurface /> };
  if (linear && nf === 2 && trace.byRole("input", "x")?.values) {
    if (hasLoss && phase !== "forward") out.push(loss, prob);
    else out.push(prob, ...(hasLoss ? [loss] : []));
  } else if (linear && nf === 1 && hasLoss) {
    out.push(loss);
  }
  out.push({ id: "tensors", title: "Tensor stack", render: () => <TensorStack /> });
  return out;
}

/** The honest mapping caption for a scene. */
export function sceneCaption(id: SceneId, trace: ReturnType<typeof useAppState>["trace"]): string {
  if (id === "probability") {
    return "Height = predicted probability p(x₁, x₂) from this step's parameters. Colour: class-0 hue at p = 0 to class-1 hue at p = 1. Position = the two input features. Spheres: traced samples at their predicted height (colour = label); the black line on the surface is p = 0.5. Faded floor dots: the whole dataset.";
  }
  if (id === "loss") {
    const note = trace.context("loss_landscape")?.note ?? "";
    return `Height and colour = mean loss on the recorded grid (low to high). Position = the two parameters. Marker: the parameters current at this step; dots and line: the path over iterations. Marker and path heights are bilinear interpolations of the grid, not separately computed losses. ${note}`;
  }
  return "One grid per forward-step output, in step order along the depth axis. Each cell is one value: colour = the 2D diverging scale (blue positive, orange negative, intensity relative to the tensor's max |value|). Position = row/column in the tensor. Current step highlighted, others dimmed. Click a cell to select it.";
}
