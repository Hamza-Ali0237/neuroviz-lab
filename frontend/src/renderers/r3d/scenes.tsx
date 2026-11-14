import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { Line } from "@react-three/drei";
import { useAppState, useDispatch } from "../../state/store";
import { fmt } from "../../format";
import { flatIndex, unflatIndex } from "../../trace/load";
import { mlpView, probGridAt } from "../mlp";
import { NEG_RGB, POS_RGB } from "../../format";
import { Axes, Label, SIZE, Surface, bilinear, cellColor, orbitPosition, span, useTheme } from "./common";

export type SceneId = "probability" | "loss" | "tensors" | "network" | "surface" | "regions";

export interface SceneDef {
  id: SceneId;
  title: string;
  /** Static default camera direction (the distance is fitted to the canvas); orbit controls reset to it. */
  camera: [number, number, number];
  render: () => JSX.Element;
}

/** 35° elevation, 45° azimuth: both floor axes read left-to-right and the surface is not hidden by itself. */
export const DEFAULT_CAMERA = orbitPosition(35, 45, 1);

const GRID = 41;

/**
 * p(x1, x2) as a height field. Logistic regression: computed from the parameters current
 * at this step. MLP: the recorded probability grid current at this step (prob_grid_k).
 * Height = probability; traced samples as spheres at their predicted height.
 */
export function ProbabilitySurface() {
  const { trace, stepIndex, sampleIndex, selection } = useAppState();
  const dispatch = useDispatch();
  const theme = useTheme();
  const x = trace.byRole("input", "x")!;
  const y = trace.byRole("label");
  const ctxX = trace.context("dataset_X");
  const ctxY = trace.context("dataset_y");
  const n = trace.raw.mode.n_samples;
  const names = trace.raw.dataset?.features ?? ["x1", "x2"];
  const linear = trace.raw.model.family === "linear";
  const params = trace.paramsAt(stepIndex);
  const w = params.get("w");
  const b = params.get("b");
  const view = linear ? null : mlpView(trace, stepIndex, sampleIndex);
  const g = linear ? null : probGridAt(trace, view?.reached ?? 0);
  // Axes: LR spans the data range; the MLP grid has its own axes.
  const all = [...x.values!, ...(ctxX?.values ?? [])];
  const lo = Math.floor(Math.min(-2.5, ...all) - 0.5);
  const hi = Math.ceil(Math.max(2.5, ...all) + 0.5);
  const [lo1, hi1, lo2, hi2] = g ? [g.ax1[0], g.ax1[g.ax1.length - 1], g.ax2[0], g.ax2[g.ax2.length - 1]] : [lo, hi, lo, hi];
  const s1 = span(lo1, hi1);
  const s2 = span(lo2, hi2);
  const sy = (p: number) => p * (SIZE / 2);
  const [w1, w2] = w?.values ?? [0, 0];
  const b0 = b?.values?.[0] ?? 0;
  const pLR = (x1: number, x2: number) => 1 / (1 + Math.exp(-(w1 * x1 + w2 * x2 + b0)));
  const ax1 = useMemo(() => (g ? Array.from(g.ax1) : Array.from({ length: GRID }, (_, k) => lo + ((hi - lo) * k) / (GRID - 1))), [g?.ax1, lo, hi]); // eslint-disable-line react-hooks/exhaustive-deps
  const ax2 = useMemo(() => (g ? Array.from(g.ax2) : ax1), [g?.ax2, ax1]); // eslint-disable-line react-hooks/exhaustive-deps
  const gridId = g?.grid.id ?? "";
  const pAt = useMemo(
    () => (i: number, j: number) => (g ? g.grid.values![i * ax2.length + j] : pLR(ax1[i], ax2[j])),
    [gridId, ax1, ax2, w1, w2, b0], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const xs = useMemo(() => ax1.map(s1), [ax1, s1]);
  const zs = useMemo(() => ax2.map(s2), [ax2, s2]);
  const heights = useMemo(() => (i: number, j: number) => sy(pAt(i, j)), [pAt]); // eslint-disable-line react-hooks/exhaustive-deps
  const color = useMemo(() => (i: number, j: number) => theme.pt0.clone().lerp(theme.pt1, pAt(i, j)).lerp(theme.panel, 0.35), [pAt, theme]);
  // Decision line w1·x1 + w2·x2 + b = 0 at height p = 0.5, clipped to the data range (LR only).
  const line = useMemo(() => {
    const pts: [number, number][] = [];
    if (!linear) return pts;
    const push = (x1: number, x2: number) => x1 >= lo && x1 <= hi && x2 >= lo && x2 <= hi && pts.push([x1, x2]);
    if (Math.abs(w2) > 1e-9) for (const x1 of [lo, hi]) push(x1, -(w1 * x1 + b0) / w2);
    if (Math.abs(w1) > 1e-9) for (const x2 of [lo, hi]) push(-(w2 * x2 + b0) / w1, x2);
    return pts.slice(0, 2);
  }, [linear, w1, w2, b0, lo, hi]);
  // Sample heights: LR exact from the parameters; MLP exact from the softmax output of the
  // iteration that used these parameters, else interpolated from the grid (flagged).
  const softmax = linear
    ? null
    : trace.raw.nodes.find((nd) => nd.op === "softmax" && nd.phase === "forward" && trace.steps[nd.step].iteration === (view?.reached ?? 0));
  const probs = softmax ? trace.tensor(softmax.outputs[0]).values : null;
  const sampleP = (i: number): [number, boolean] => {
    const x1 = x.values![i * 2];
    const x2 = x.values![i * 2 + 1];
    if (linear) return [pLR(x1, x2), false];
    if (probs) return [probs[i * 2 + 1], false];
    return [bilinear(g!.grid.values!, g!.ax1, g!.ax2, x1, x2) ?? 0, true];
  };
  const selRow = selection.tensorId === x.id && selection.index ? selection.index[0] : -1;
  const cls = (l: number | null | undefined) => (l === null || l === undefined ? theme.muted : l > 0.5 ? theme.pt1 : theme.pt0);
  return (
    <group position={[0, -SIZE / 4, 0]}>
      <Axes x={{ name: names[0], lo: lo1, hi: hi1 }} z={{ name: names[1], lo: lo2, hi: hi2 }} y={{ name: "p", lo: 0, hi: 1 }} colors={theme} />
      <Surface xs={xs} zs={zs} heights={heights} color={color} opacity={0.8} />
      {line.length === 2 && (
        <Line points={line.map(([x1, x2]) => [s1(x1), sy(0.5) + 0.01, s2(x2)] as [number, number, number])} color={theme.fg} lineWidth={2} />
      )}
      {ctxX?.values &&
        ctxX.shape[1] === 2 &&
        Array.from({ length: ctxX.shape[0] }, (_, k) => (
          <mesh key={k} position={[s1(ctxX.values![k * 2]), 0.03, s2(ctxX.values![k * 2 + 1])]}>
            <sphereGeometry args={[0.035, 8, 8]} />
            <meshStandardMaterial color={cls(ctxY?.values?.[k])} transparent opacity={0.35} />
          </mesh>
        ))}
      {Array.from({ length: n }, (_, i) => {
        const x1 = x.values![i * 2];
        const x2 = x.values![i * 2 + 1];
        const [pi, approx] = sampleP(i);
        const current = i === sampleIndex;
        return (
          <group key={i}>
            <mesh position={[s1(x1), 0.02, s2(x2)]}>
              <sphereGeometry args={[0.06, 10, 10]} />
              <meshStandardMaterial color={cls(y?.values?.[i])} transparent opacity={0.45} />
            </mesh>
            <Line points={[[s1(x1), 0, s2(x2)], [s1(x1), sy(pi), s2(x2)]]} color={theme.muted} lineWidth={1} dashed dashSize={0.08} gapSize={0.06} />
            <mesh
              position={[s1(x1), sy(pi), s2(x2)]}
              onClick={(e) => {
                e.stopPropagation();
                dispatch({ type: "sample", index: i });
                dispatch({ type: "select", selection: { tensorId: x.id, index: [i, 0] } });
              }}
            >
              <sphereGeometry args={[current ? 0.11 : 0.08, 16, 16]} />
              <meshStandardMaterial color={cls(y?.values?.[i])} emissive={i === selRow ? theme.sel : current ? theme.fg : new THREE.Color(0)} emissiveIntensity={0.5} />
            </mesh>
            {current && <Label at={[s1(x1), sy(pi) + 0.25, s2(x2)]}>{`sample ${i}: p ${approx ? "≈" : "="} ${fmt(pi)}${approx ? " (from grid)" : ""}`}</Label>}
          </group>
        );
      })}
    </group>
  );
}

/**
 * MLP network in space: layers as rows of spheres (colour = activation for the selected
 * sample, dim until the layer's step is reached), connections as lines coloured by weight
 * sign and thickened by magnitude (gradients on backward steps).
 */
export function NetworkScene() {
  const { trace, stepIndex, sampleIndex, selection } = useAppState();
  const dispatch = useDispatch();
  const theme = useTheme();
  const view = mlpView(trace, stepIndex, sampleIndex);
  if (!view) return null;
  const { layers, edges } = view;
  const lx = (l: number) => (l - (layers.length - 1) / 2) * 1.7;
  const nz = (size: number, j: number) => (j - (size - 1) / 2) * 0.8;
  const ny = 0.9;
  const pos = (l: number, j: number): [number, number, number] => [lx(l), ny, nz(layers[l].size, j)];
  const pos3 = (c: [number, number, number]) => new THREE.Color(c[0] / 255, c[1] / 255, c[2] / 255);
  return (
    <group position={[0, -SIZE / 4, 0]}>
      <gridHelper args={[SIZE, 8, theme.border, theme.border]} />
      {edges.map((e, l) => {
        const t = e.tensor;
        const [nIn, nOut] = t.shape;
        const absMax = Math.max(Math.abs(t.stats.min), Math.abs(t.stats.max), 1e-12);
        const lines: JSX.Element[] = [];
        for (let i = 0; i < nIn; i++) {
          for (let j = 0; j < nOut; j++) {
            const v = t.values![i * nOut + j];
            const a = Math.abs(v) / absMax;
            lines.push(
              <Line
                key={`${l}-${i}-${j}`}
                points={[pos(l, i), pos(l + 1, j)]}
                color={pos3(v >= 0 ? POS_RGB : NEG_RGB)}
                lineWidth={0.5 + 4 * a}
                transparent
                opacity={0.3 + 0.7 * a}
                onClick={(ev) => {
                  ev.stopPropagation();
                  dispatch({ type: "select", selection: { tensorId: t.id, index: [i, j] } });
                }}
              />,
            );
          }
        }
        return (
          <group key={l}>
            {lines}
            <Label at={[(lx(l) + lx(l + 1)) / 2, ny + 1.2, 0]} muted>{`${e.grad ? "∂L/∂" : ""}${e.name} (${t.shape.join("×")})`}</Label>
          </group>
        );
      })}
      {layers.map((layer, l) =>
        Array.from({ length: layer.size }, (_, j) => {
          const v = layer.values?.[j] ?? null;
          const lit = v !== null;
          const sel = selection.tensorId === layer.tensorId && selection.index?.[0] === sampleIndex && selection.index?.[1] === j;
          return (
            <group key={`${l}-${j}`}>
              <mesh
                position={pos(l, j)}
                onClick={(ev) => {
                  ev.stopPropagation();
                  if (layer.tensorId) dispatch({ type: "select", selection: { tensorId: layer.tensorId, index: [sampleIndex, j] } });
                }}
              >
                <sphereGeometry args={[0.2, 20, 20]} />
                <meshStandardMaterial
                  color={lit ? cellColor(v!, layer.absMax, theme.panel) : theme.panel}
                  transparent={!lit}
                  opacity={lit ? 1 : 0.35}
                  emissive={sel ? theme.sel : new THREE.Color(0)}
                  emissiveIntensity={0.7}
                />
              </mesh>
              {lit && (
                <Label at={[lx(l), ny - 0.42, nz(layer.size, j)]} title={layer.tensorId ? `${trace.label(layer.tensorId)}[${sampleIndex}, ${j}] = ${v}` : undefined}>
                  <span className="nvz-3d-value">{fmt(v!)}</span>
                </Label>
              )}
            </group>
          );
        }),
      )}
      {layers.map((layer, l) => (
        <Label key={l} at={[lx(l), 0, SIZE / 2 + 0.3]}>{layer.label}</Label>
      ))}
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
  const lossPath = trace.context("loss_path")?.values ?? null;
  const min = Math.min(grid.stats.min, ...(lossPath ?? []));
  const max = Math.max(grid.stats.max, ...(lossPath ?? []));
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
  const reached = updates.filter((e) => e.step <= stepIndex).length;
  // Heights: the exact loss recorded for path point k (loss_path); older traces without it
  // fall back to interpolating the grid.
  const toScene = ([a, b]: [number, number], k: number): [number, number, number] => {
    const l = lossPath && k < lossPath.length ? lossPath[k] : bilinear(grid.values!, ax0, ax1, a, b);
    return [s0(a), l === null ? 0 : sy(l) + 0.02, s1(b)];
  };
  const scenePath = path.map(toScene);
  const current = scenePath[Math.min(reached, scenePath.length - 1)];
  const currentLoss = lossPath && reached < lossPath.length ? lossPath[reached] : null;
  return (
    <group position={[0, -SIZE / 4, 0]}>
      <Axes x={{ name: n0, lo: lo0, hi: hi0 }} z={{ name: n1, lo: lo1, hi: hi1 }} y={{ name: "loss", lo: min, hi: max }} colors={theme} />
      <Surface xs={xs} zs={zs} heights={heights} color={color} opacity={0.9} />
      {scenePath.length > 1 && <Line points={scenePath} color={theme.muted} lineWidth={1.5} dashed dashSize={0.1} gapSize={0.06} />}
      {reached > 0 && <Line points={scenePath.slice(0, reached + 1)} color={theme.fg} lineWidth={3} />}
      {scenePath.length > 0 && <Label at={[scenePath[0][0], scenePath[0][1] - 0.2, scenePath[0][2]]} muted>start</Label>}
      {scenePath.map((pt, k) => (
        <mesh key={k} position={pt}>
          <sphereGeometry args={[0.045, 12, 12]} />
          <meshStandardMaterial color={k <= reached ? theme.fg : theme.muted} />
        </mesh>
      ))}
      <mesh position={current}>
        <sphereGeometry args={[0.09, 16, 16]} />
        <meshStandardMaterial color={theme.accent} emissive={theme.accent} emissiveIntensity={0.4} />
      </mesh>
      <Label at={[current[0], current[1] + 0.35, current[2]]}>
        {`${n0} = ${fmt(path[Math.min(reached, path.length - 1)][0])}, ${n1} = ${fmt(path[Math.min(reached, path.length - 1)][1])}${currentLoss === null ? "" : ` · loss = ${fmt(currentLoss)}`}`}
      </Label>
    </group>
  );
}

/** One tensor as an instanced grid of cells; a wireframe box marks the selected cell. */
function CellPlane({ t, step, z, cell, gap, active }: {
  t: { id: string; shape: number[]; size: number; values: Float32Array | null; stats: { min: number; max: number } };
  step: number;
  z: number;
  cell: number;
  gap: number;
  active: boolean;
}) {
  const { trace, selection, sampleIndex } = useAppState();
  const dispatch = useDispatch();
  const theme = useTheme();
  const ref = useRef<THREE.InstancedMesh>(null);
  const cols = t.shape.length ? t.shape[t.shape.length - 1] : 1;
  const rows = t.size / cols;
  const pitch = cell + gap;
  const w = cols * pitch;
  const h = rows * pitch;
  const absMax = Math.max(Math.abs(t.stats.min), Math.abs(t.stats.max), 1e-12);
  const perSample = t.shape.length >= 1 && t.shape[0] === trace.raw.mode.n_samples && trace.raw.mode.n_samples > 1;
  const selectedFlat = selection.tensorId === t.id && selection.index ? flatIndex(t.shape, selection.index) : -1;
  const place = (flat: number): [number, number, number] => [(flat % cols - (cols - 1) / 2) * pitch, h - (Math.floor(flat / cols) + 0.5) * pitch, 0];
  const values = t.values;
  useLayoutEffect(() => {
    const m = ref.current;
    if (!m || !values) return;
    const o = new THREE.Object3D();
    for (let flat = 0; flat < values.length; flat++) {
      const [px, py] = place(flat);
      const isSample = perSample && unflatIndex(t.shape, flat)[0] === sampleIndex;
      o.position.set(px, py, 0);
      o.scale.set(1, 1, flat === selectedFlat ? 0.8 : 0.3);
      o.updateMatrix();
      m.setMatrixAt(flat, o.matrix);
      const c = cellColor(values[flat], absMax, theme.panel);
      if (flat === selectedFlat) c.lerp(theme.sel, 0.6);
      else if (isSample && active) c.lerp(theme.accent, 0.25);
      m.setColorAt(flat, c);
    }
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
  }); // every render: cheap (one loop), and selection/sample/theme all feed the colours
  if (!values) return null;
  const select = (flat: number) => {
    const idx = unflatIndex(t.shape, flat);
    dispatch({ type: "select", selection: { tensorId: t.id, index: idx } });
    if (perSample) dispatch({ type: "sample", index: idx[0] });
  };
  // Value labels: the whole plane when it is current and the cells are big enough to hold
  // the text; otherwise only the selected cell, so labels never spill over plane edges.
  const labelAll = active && values.length <= 64 && cell >= 0.3;
  const labelled = labelAll ? Array.from(values, (_, i) => i) : selectedFlat >= 0 ? [selectedFlat] : [];
  return (
    <group position={[0, 0.2, z]}>
      <Label at={[0, h + 0.3, 0]} muted={!active}>
        {`${trace.steps[step].title} · ${trace.label(t.id)} (${t.shape.join(", ")})`}
      </Label>
      <instancedMesh
        key={`${t.id}:${values.length}`}
        ref={ref}
        args={[undefined, undefined, values.length]}
        onClick={(e) => {
          e.stopPropagation();
          if (e.instanceId !== undefined) select(e.instanceId);
        }}
      >
        <boxGeometry args={[cell, cell, cell]} />
        <meshStandardMaterial transparent opacity={active ? 1 : 0.3} />
      </instancedMesh>
      {selectedFlat >= 0 && (
        <mesh position={place(selectedFlat)}>
          <boxGeometry args={[cell * 1.08, cell * 1.08, cell * 0.85]} />
          <meshBasicMaterial color={theme.fg} wireframe />
        </mesh>
      )}
      {labelled.map((flat) => {
        const [px, py] = place(flat);
        const idx = unflatIndex(t.shape, flat);
        return (
          <Label key={flat} at={[px, py, cell * 0.5]} onClick={() => select(flat)} title={`${trace.label(t.id)}[${idx.join(", ")}] = ${values[flat]}`}>
            <span className="nvz-3d-value">{fmt(values[flat])}</span>
          </Label>
        );
      })}
      <Line points={[[-w / 2, 0, 0], [w / 2, 0, 0], [w / 2, h, 0], [-w / 2, h, 0], [-w / 2, 0, 0]]} color={active ? theme.fg : theme.border} lineWidth={active ? 2 : 1} />
    </group>
  );
}

/**
 * Generic view for any trace: each forward step's output tensor as a grid of coloured cells,
 * stacked along the depth axis in step order. The current step's tensors are highlighted.
 * Instanced: a CNN-sized trace (tens of tensors, hundreds of values each) stays one draw
 * call per tensor.
 */
export function TensorStack() {
  const { trace, stepIndex } = useAppState();
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
  const cell = Math.min(0.6, (SIZE * 0.9) / maxDim / 1.2);
  const gap = cell * 0.2;
  const depth = outs.length > 1 ? Math.min(1.4, SIZE / (outs.length - 1)) : 0;
  const z0 = (-(outs.length - 1) * depth) / 2;
  return (
    <group position={[0, -SIZE / 4, 0]}>
      <gridHelper args={[SIZE, 8, theme.border, theme.border]} />
      {outs.map(({ step, tensor: t }, k) => (
        <CellPlane key={t.id} t={t} step={step} z={z0 + k * depth} cell={cell} gap={gap} active={step === stepIndex} />
      ))}
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
  const loss: SceneDef = { id: "loss", title: "Loss landscape", camera: DEFAULT_CAMERA, render: () => <LossLandscape /> };
  const prob: SceneDef = { id: "probability", title: "Probability surface", camera: DEFAULT_CAMERA, render: () => <ProbabilitySurface /> };
  if (linear && nf === 2 && trace.byRole("input", "x")?.values) {
    if (hasLoss && phase !== "forward") out.push(loss, prob);
    else out.push(prob, ...(hasLoss ? [loss] : []));
  } else if (linear && nf === 1 && hasLoss) {
    out.push(loss);
  }
  if (trace.raw.model.family === "mlp" && mlpView(trace, 0, 0)) {
    out.push({ id: "network", title: "Network", camera: orbitPosition(28, 20, 1), render: () => <NetworkScene /> });
    if (probGridAt(trace, 0)) out.push({ id: "surface", title: "Probability surface", camera: DEFAULT_CAMERA, render: () => <ProbabilitySurface /> });
  }
  out.push({ id: "tensors", title: "Tensor stack", camera: DEFAULT_CAMERA, render: () => <TensorStack /> });
  return out;
}

/** The honest mapping caption for a scene. */
export function sceneCaption(id: SceneId, trace: ReturnType<typeof useAppState>["trace"]): string {
  if (id === "network") {
    return "Spheres = neurons, one row per layer (input, hidden, output), colour = the value for the selected sample on the diverging scale (blue positive, orange negative, per layer); dim spheres are layers not computed yet at this step. Lines = weights: blue positive, orange negative, thickness and opacity ∝ |weight|; on backward steps they show ∂L/∂W instead once computed. Click a sphere or a line to select the value.";
  }
  if (id === "surface") {
    const note = trace.context("prob_grid_0")?.note ?? "";
    return `Height and colour = probability of class 1 from the recorded grid current at this step (class-0 hue at 0, class-1 hue at 1). Position = the two input features. Spheres: traced samples at their predicted probability (exact softmax output when this step's forward pass has run; otherwise interpolated from the grid and marked ≈). Faded floor dots: the whole dataset. ${note}`;
  }
  if (id === "probability") {
    return "Height = predicted probability p(x₁, x₂) from this step's parameters. Colour: class-0 hue at p = 0 to class-1 hue at p = 1. Position = the two input features. Spheres: traced samples at their predicted height (colour = label); the black line on the surface is p = 0.5. Faded floor dots: the whole dataset.";
  }
  if (id === "loss") {
    const note = trace.context("loss_landscape")?.note ?? "";
    const exact = trace.context("loss_path")
      ? "Marker and path heights are the exact losses recorded at each visited parameter set (loss_path), computed with the same samples and loss function."
      : "Marker and path heights are bilinear interpolations of the grid, not separately computed losses.";
    return `Height and colour = mean loss on the recorded grid (low to high). Position = the two parameters. Marker: the parameters current at this step; dots and line: the path over iterations. ${exact} ${note}`;
  }
  return "One grid per forward-step output, in step order along the depth axis. Each cell is one value: colour = the 2D diverging scale (blue positive, orange negative, intensity relative to the tensor's max |value|). Position = row/column in the tensor. Current step highlighted, others dimmed. Click a cell to select it.";
}
