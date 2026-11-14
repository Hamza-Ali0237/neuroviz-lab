import { useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Line } from "@react-three/drei";
import { useAppState, useDispatch } from "../../state/store";
import { fmt } from "../../format";
import { flatIndex, unflatIndex } from "../../trace/load";
import { mlpView, probGridAt } from "../mlp";
import { dataBounds, regions, rule, treeAt } from "../tree";
import { NEG_RGB, POS_RGB } from "../../format";
import { Axes, Label, SIZE, Surface, bilinear, cellColor, orbitPosition, span, useTheme } from "./common";
import { cnnNode } from "../r2d/CNNScene";
import { attentionNode } from "../r2d/TransformerScene";
import { tokenLabels } from "../tokens";

export type SceneId = "probability" | "loss" | "tensors" | "network" | "surface" | "regions" | "conv" | "attention";

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
              {/* Outline shell: a slightly larger back-face sphere, so a zero-valued (panel-coloured) neuron never vanishes. */}
              <mesh position={pos(l, j)} scale={1.07}>
                <sphereGeometry args={[0.2, 20, 20]} />
                <meshBasicMaterial color={lit ? theme.fg : theme.border} side={THREE.BackSide} transparent={!lit} opacity={lit ? 1 : 0.5} />
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
      {scenePath.length > 1 && <Line points={scenePath} color={theme.muted} lineWidth={2.5} dashed dashSize={0.1} gapSize={0.06} />}
      {reached > 0 && <Line points={scenePath.slice(0, reached + 1)} color={theme.fg} lineWidth={5} />}
      {scenePath.map((pt, k) => {
        const l = lossPath && k < lossPath.length ? lossPath[k] : null;
        const isCurrent = k === Math.min(reached, scenePath.length - 1);
        // Iteration dots are numbered; their exact loss sits below the dot, the current marker's label above it.
        return (
          <group key={k}>
            <mesh position={pt}>
              <sphereGeometry args={[isCurrent ? 0.1 : 0.06, 16, 16]} />
              <meshStandardMaterial color={isCurrent ? theme.accent : k <= reached ? theme.fg : theme.muted} emissive={isCurrent ? theme.accent : new THREE.Color(0)} emissiveIntensity={0.4} />
            </mesh>
            <Label at={[pt[0], pt[1] - 0.26, pt[2]]} muted={!isCurrent}>
              {`${k}${k === 0 ? " start" : ""}${l === null ? "" : ` · loss ${fmt(l)}`}`}
            </Label>
          </group>
        );
      })}
      <Label at={[current[0], current[1] + 0.4, current[2]]}>
        {`${n0} = ${fmt(path[Math.min(reached, path.length - 1)][0])}, ${n1} = ${fmt(path[Math.min(reached, path.length - 1)][1])}${currentLoss === null ? "" : ` · loss = ${fmt(currentLoss)}`}`}
      </Label>
    </group>
  );
}

/** One tensor as an instanced grid of cells; a wireframe box marks the selected cell. */
export function CellPlane({ t, step, z, cell, gap, active, prefix = [], title, at = [0, 0.2] }: {
  t: { id: string; shape: number[]; size: number; values: Float32Array | null; stats: { min: number; max: number } };
  step: number;
  z: number;
  cell: number;
  gap: number;
  active: boolean;
  /** Fixed leading indices: draw only the block t[...prefix] (its last two dims as rows × cols). */
  prefix?: number[];
  /** Label above the plane (default: step title · tensor name (shape)). */
  title?: string;
  /** x, y position of the plane's bottom centre. */
  at?: [number, number];
}) {
  const { trace, selection, sampleIndex } = useAppState();
  const dispatch = useDispatch();
  const theme = useTheme();
  const ref = useRef<THREE.InstancedMesh>(null);
  const rest = t.shape.slice(prefix.length);
  const cols = rest.length ? rest[rest.length - 1] : 1;
  const count = rest.reduce((a, b) => a * b, 1);
  const rows = count / cols;
  const base = prefix.length ? flatIndex(t.shape, [...prefix, ...rest.map(() => 0)]) : 0;
  const pitch = cell + gap;
  const w = cols * pitch;
  const h = rows * pitch;
  const absMax = Math.max(Math.abs(t.stats.min), Math.abs(t.stats.max), 1e-12);
  const perSample = !prefix.length && t.shape.length >= 1 && t.shape[0] === trace.raw.mode.n_samples && trace.raw.mode.n_samples > 1;
  const selectedFlat = selection.tensorId === t.id && selection.index ? flatIndex(t.shape, selection.index) - base : -1;
  const place = (flat: number): [number, number, number] => [(flat % cols - (cols - 1) / 2) * pitch, h - (Math.floor(flat / cols) + 0.5) * pitch, 0];
  const values = t.values;
  useLayoutEffect(() => {
    const m = ref.current;
    if (!m || !values) return;
    const o = new THREE.Object3D();
    for (let flat = 0; flat < count; flat++) {
      const [px, py] = place(flat);
      const isSample = perSample && unflatIndex(t.shape, flat)[0] === sampleIndex;
      o.position.set(px, py, 0);
      o.scale.set(1, 1, flat === selectedFlat ? 0.8 : 0.3);
      o.updateMatrix();
      m.setMatrixAt(flat, o.matrix);
      const c = cellColor(values[base + flat], absMax, theme.panel);
      if (flat === selectedFlat) c.lerp(theme.sel, 0.6);
      else if (isSample && active) c.lerp(theme.accent, 0.25);
      m.setColorAt(flat, c);
    }
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
  }); // every render: cheap (one loop), and selection/sample/theme all feed the colours
  if (!values) return null;
  const select = (flat: number) => {
    const idx = unflatIndex(t.shape, base + flat);
    dispatch({ type: "select", selection: { tensorId: t.id, index: idx } });
    if (perSample) dispatch({ type: "sample", index: idx[0] });
  };
  // Value labels: the whole plane when it is current and the cells are big enough to hold
  // the text; otherwise only the selected cell, so labels never spill over plane edges.
  const labelAll = active && count <= 64 && cell >= 0.3;
  const labelled = labelAll ? Array.from({ length: count }, (_, i) => i) : selectedFlat >= 0 && selectedFlat < count ? [selectedFlat] : [];
  return (
    <group position={[at[0], at[1], z]}>
      <Label at={[0, h + 0.3, 0]} muted={!active}>
        {title ?? `${trace.steps[step].title} · ${trace.label(t.id)} (${t.shape.join(", ")})`}
      </Label>
      <instancedMesh
        key={`${t.id}:${prefix.join(",")}:${count}`}
        ref={ref}
        args={[undefined, undefined, count]}
        onClick={(e) => {
          e.stopPropagation();
          if (e.instanceId !== undefined) select(e.instanceId);
        }}
      >
        <boxGeometry args={[cell, cell, cell]} />
        <meshStandardMaterial transparent opacity={active ? 1 : 0.3} />
      </instancedMesh>
      {selectedFlat >= 0 && selectedFlat < count && (
        <mesh position={place(selectedFlat)}>
          <boxGeometry args={[cell * 1.08, cell * 1.08, cell * 0.85]} />
          <meshBasicMaterial color={theme.fg} wireframe />
        </mesh>
      )}
      {labelled.map((flat) => {
        const [px, py] = place(flat);
        const idx = unflatIndex(t.shape, base + flat);
        return (
          <Label key={flat} at={[px, py, cell * 0.5]} onClick={() => select(flat)} title={`${trace.label(t.id)}[${idx.join(", ")}] = ${values[base + flat]}`}>
            <span className="nvz-3d-value">{fmt(values[base + flat])}</span>
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

/**
 * Decision tree in feature space: one box per leaf (or not-yet-split node). Footprint = the
 * region's bounds, height and colour = predicted probability of class 1 at that node;
 * dataset points on the floor.
 */
export function RegionsScene() {
  const { trace, stepIndex, sampleIndex, selection } = useAppState();
  const dispatch = useDispatch();
  const theme = useTheme();
  const [hovered, setHovered] = useState<number | null>(null);
  const tree = treeAt(trace, stepIndex);
  const bounds = dataBounds(trace);
  const x = trace.byRole("input", "x");
  const y = trace.byRole("label");
  const isTrain = trace.raw.kind === "train";
  const ctxX = trace.context("dataset_X") ?? (isTrain ? x : null); // train: the dataset is x itself
  const ctxY = trace.context("dataset_y") ?? (isTrain ? y : null);
  if (!tree || !bounds) return null;
  const nf = bounds.length;
  const names = trace.raw.dataset?.features ?? ["x1", "x2"];
  const [lo1, hi1] = bounds[0];
  const [lo2, hi2] = nf > 1 ? bounds[1] : [0, 1];
  const s1 = span(lo1, hi1);
  const s2 = span(lo2, hi2);
  const yTop = SIZE / 4; // half the cube: boxes stay low so the dataset on the floor remains visible
  const sy = (p: number) => Math.max(0.02, p * yTop); // same scale as the Axes y range 0..1
  const regs = regions(tree, bounds);
  const splitting = tree.split?.node ?? null;
  const path = tree.paths[sampleIndex] ?? [];
  const here = path.length ? path[path.length - 1] : null;
  const traced = isTrain ? 0 : trace.raw.mode.n_samples;
  const cls = (l: number | null | undefined) => (l === null || l === undefined ? theme.muted : l > 0.5 ? theme.pt1 : theme.pt0);
  const selRow = x && selection.tensorId === x.id && selection.index ? selection.index[0] : -1;
  return (
    <group position={[0, -SIZE / 4, 0]}>
      <Axes x={{ name: names[0], lo: lo1, hi: hi1 }} z={{ name: nf > 1 ? names[1] : "", lo: lo2, hi: hi2 }} y={{ name: "p(class 1)", lo: 0, hi: 1 }} colors={theme} yHeight={yTop} />
      {regs.map(({ node, bounds: b }) => {
        const w = s1(b[0][1]) - s1(b[0][0]);
        const d = nf > 1 ? s2(b[1][1]) - s2(b[1][0]) : SIZE;
        const p1 = node.n_samples ? (node.counts[1] ?? 0) / node.n_samples : 0; // probability of class 1
        const hgt = sy(p1);
        const cx = (s1(b[0][0]) + s1(b[0][1])) / 2;
        const cz = nf > 1 ? (s2(b[1][0]) + s2(b[1][1])) / 2 : 0;
        const hot = node.id === splitting || node.id === here;
        const colour = theme.pt0.clone().lerp(theme.pt1, p1).lerp(theme.panel, 0.3);
        return (
          <group key={node.id}>
            <mesh
              position={[cx, hgt / 2, cz]}
              onClick={(e) => { e.stopPropagation(); if (x) dispatch({ type: "select", selection: { tensorId: x.id, index: [sampleIndex, 0] } }); }}
              onPointerOver={(e) => { e.stopPropagation(); setHovered(node.id); }}
              onPointerOut={() => setHovered((h) => (h === node.id ? null : h))}
            >
              <boxGeometry args={[Math.max(0.01, w - 0.02), hgt, Math.max(0.01, d - 0.02)]} />
              <meshStandardMaterial color={colour} transparent opacity={node.pending ? 0.12 : hot ? 0.4 : 0.22} depthWrite={false} emissive={hot ? theme.accent : new THREE.Color(0)} emissiveIntensity={0.35} />
            </mesh>
            <Line points={[[cx - w / 2, hgt, cz - d / 2], [cx + w / 2, hgt, cz - d / 2], [cx + w / 2, hgt, cz + d / 2], [cx - w / 2, hgt, cz + d / 2], [cx - w / 2, hgt, cz - d / 2]]} color={hot ? theme.accent : theme.fg} lineWidth={hot ? 2.5 : 1} transparent opacity={hot ? 0.9 : 0.45} />
            {(hot || hovered === node.id) && <Label at={[cx, hgt + 0.22, cz]}>{`node ${node.id}: p₁ = ${fmt(p1)}${node.feature !== null ? ` · ${rule(node)}` : ""}`}</Label>}
          </group>
        );
      })}
      {ctxX?.values &&
        Array.from({ length: ctxX.shape[0] }, (_, k) => (
          <mesh key={k} position={[s1(ctxX.values![k * nf]), 0.03, nf > 1 ? s2(ctxX.values![k * nf + 1]) : ((ctxY?.values?.[k] ?? 0.5) - 0.5) * 2]}>
            <sphereGeometry args={[traced ? 0.035 : 0.05, 8, 8]} />
            <meshStandardMaterial color={cls(ctxY?.values?.[k])} transparent opacity={traced ? 0.35 : 0.8} />
          </mesh>
        ))}
      {x?.values &&
        Array.from({ length: traced }, (_, i) => {
          const current = i === sampleIndex;
          const px = s1(x.values![i * nf]);
          const pz = nf > 1 ? s2(x.values![i * nf + 1]) : ((y?.values?.[i] ?? 0.5) - 0.5) * 2;
          return (
            <group key={i}>
              <mesh position={[px, 0.08, pz]} onClick={(e) => { e.stopPropagation(); dispatch({ type: "sample", index: i }); dispatch({ type: "select", selection: { tensorId: x.id, index: [i, 0] } }); }}>
                <sphereGeometry args={[current ? 0.11 : 0.08, 16, 16]} />
                <meshStandardMaterial color={cls(y?.values?.[i])} emissive={i === selRow ? theme.sel : current ? theme.fg : new THREE.Color(0)} emissiveIntensity={0.5} />
              </mesh>
              {current && <Label at={[px, 0.4, pz]}>{`sample ${i}`}</Label>}
            </group>
          );
        })}
    </group>
  );
}


/**
 * CNN convolution in space: the input's channels as stacked planes, the output feature maps
 * as stacked planes behind them, and a translucent box around the 3×3×3 receptive-field
 * volume that produced the selected output cell (default: map 0, cell (0, 0)).
 */
export function ConvStack() {
  const { trace, stepIndex, selection, sampleIndex } = useAppState();
  const theme = useTheme();
  const node = trace.raw.nodes.find((n) => n.op === "conv2d" && n.phase === "forward") ?? null;
  if (!node) return null;
  const [xId, wId] = node.inputs;
  const outId = node.outputs[0];
  const x = trace.tensor(xId);
  const out = trace.tensor(outId);
  const [, C, H, W] = x.shape;
  const [, O, HO, WO] = out.shape;
  const [, , kh, kw] = trace.tensor(wId).shape;
  const s = (node.attrs.stride as number | undefined) ?? 1;
  const p = (node.attrs.padding as number | undefined) ?? 0;
  const n = Math.min(sampleIndex, x.shape[0] - 1);
  const selOut = selection.tensorId === outId && selection.index?.length === 4 ? selection.index : [n, 0, 0, 0];
  const [, o, i, j] = selOut;
  const cell = Math.min(0.42, (SIZE * 0.9) / Math.max(H, W) / 1.2);
  const gap = cell * 0.2;
  const pitch = cell + gap;
  const planeH = H * pitch;
  const depth = 0.75;
  const zIn = (c: number) => -SIZE / 2 + 0.6 + c * depth;
  const zOut = (k: number) => zIn(C - 1) + 1.6 + k * depth;
  const active = trace.steps[stepIndex].phase === "forward" && stepIndex >= node.step;
  // Receptive field rows/cols (clipped to the image; padding cells are outside it).
  const r0 = Math.max(0, i * s - p);
  const r1 = Math.min(H - 1, i * s - p + kh - 1);
  const c0 = Math.max(0, j * s - p);
  const c1 = Math.min(W - 1, j * s - p + kw - 1);
  const cx = ((c0 + c1) / 2 - (W - 1) / 2) * pitch;
  const cy = 0.2 + planeH - ((r0 + r1) / 2 + 0.5) * pitch;
  const cz = (zIn(0) + zIn(C - 1)) / 2;
  const outPitch = pitch;
  const outH = HO * outPitch;
  const ox = (j - (WO - 1) / 2) * outPitch;
  const oy = 0.2 + outH - (i + 0.5) * outPitch;
  return (
    <group position={[0, -SIZE / 4, 0]}>
      <gridHelper args={[SIZE, 8, theme.border, theme.border]} />
      {Array.from({ length: C }, (_, c) => (
        <CellPlane key={`x${c}`} t={x} step={node.step} z={zIn(c)} cell={cell} gap={gap} active={active} prefix={[n, c]} title={`input channel ${c}${c === 0 ? ` (sample ${n}, ${H}×${W})` : ""}`} />
      ))}
      {Array.from({ length: O }, (_, k) => (
        <CellPlane key={`z${k}`} t={out} step={node.step} z={zOut(k)} cell={cell} gap={gap} active={active && stepIndex === node.step} prefix={[n, k]} title={`output map ${k}${k === o ? " (selected)" : ""}`} />
      ))}
      <mesh position={[cx, cy, cz]}>
        <boxGeometry args={[(c1 - c0 + 1) * pitch, (r1 - r0 + 1) * pitch, zIn(C - 1) - zIn(0) + cell]} />
        <meshStandardMaterial color={theme.accent} transparent opacity={0.28} depthWrite={false} />
      </mesh>
      <Line points={[[cx, cy, zIn(C - 1) + cell / 2], [ox, oy, zOut(o) - cell / 2]]} color={theme.accent} lineWidth={2} dashed dashSize={0.1} gapSize={0.06} />
      <Label at={[cx, cy + ((r1 - r0 + 1) * pitch) / 2 + 0.25, zIn(0)]}>{`${kh}×${kw}×${C} window for map ${o}, cell (${i}, ${j})`}</Label>
    </group>
  );
}

/**
 * Attention as a token-by-token grid of bars: height = attention weight for the selected
 * sample, token labels along both edges, the selected row highlighted.
 */
export function AttentionBars() {
  const { trace, selection, sampleIndex } = useAppState();
  const dispatch = useDispatch();
  const theme = useTheme();
  const node = attentionNode(trace);
  if (!node) return null;
  const aId = node.outputs[0];
  const a = trace.tensor(aId);
  const [nS, T] = a.shape;
  const n = Math.min(sampleIndex, nS - 1);
  const tokens = tokenLabels(trace, n) ?? Array.from({ length: T }, (_, k) => `t${k}`);
  const pitch = (SIZE * 0.8) / T;
  const bar = pitch * 0.7;
  const px = (j: number) => (j - (T - 1) / 2) * pitch;
  const pz = (i: number) => (i - (T - 1) / 2) * pitch;
  const selRow = selection.tensorId === aId && selection.index?.[0] === n ? selection.index[1] : null;
  const hTop = SIZE / 2;
  return (
    <group position={[0, -SIZE / 4, 0]}>
      <Axes x={{ name: "looks at (column)", lo: 0, hi: 1 }} z={{ name: "token (row)", lo: 0, hi: 1 }} y={{ name: "weight", lo: 0, hi: 1 }} colors={theme} ticks={false} />
      {Array.from({ length: T }, (_, i) =>
        Array.from({ length: T }, (_, j) => {
          const v = a.values![(n * T + i) * T + j];
          const hgt = Math.max(0.01, v * hTop);
          const hot = selRow === i;
          return (
            <mesh
              key={`${i}-${j}`}
              position={[px(j), hgt / 2, pz(i)]}
              onClick={(e) => { e.stopPropagation(); dispatch({ type: "select", selection: { tensorId: aId, index: [n, i, j] } }); }}
            >
              <boxGeometry args={[bar, hgt, bar]} />
              <meshStandardMaterial color={theme.pt0.clone().lerp(theme.pt1, v)} emissive={hot ? theme.accent : new THREE.Color(0)} emissiveIntensity={0.45} transparent opacity={selRow === null || hot ? 1 : 0.35} />
            </mesh>
          );
        }),
      )}
      {Array.from({ length: T }, (_, j) => (
        <Label key={`c${j}`} at={[px(j), 0, SIZE / 2 + 0.3]}>{tokens[j]}</Label>
      ))}
      {Array.from({ length: T }, (_, i) => (
        <Label key={`r${i}`} at={[-SIZE / 2 - 0.35, 0, pz(i)]} muted={selRow !== null && selRow !== i} onClick={() => dispatch({ type: "select", selection: { tensorId: aId, index: [n, i, 0] } })}>{tokens[i]}</Label>
      ))}
      {selRow !== null &&
        Array.from({ length: T }, (_, j) => {
          const v = a.values![(n * T + selRow) * T + j];
          return <Label key={`v${j}`} at={[px(j), v * hTop + 0.2, pz(selRow)]}><span className="nvz-3d-value">{fmt(v)}</span></Label>;
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
  if (trace.raw.model.family === "tree" && treeAt(trace, trace.steps.length - 1)) {
    out.push({ id: "regions", title: "Leaf regions", camera: orbitPosition(50, 40, 1), render: () => <RegionsScene /> });
  }
  if (trace.raw.model.family === "cnn" && trace.raw.nodes.some((n) => n.op === "conv2d")) {
    out.push({ id: "conv", title: "Convolution", camera: orbitPosition(22, -38, 1), render: () => <ConvStack /> });
  }
  if (trace.raw.model.family === "transformer" && attentionNode(trace)) {
    out.push({ id: "attention", title: "Attention", camera: orbitPosition(38, 30, 1), render: () => <AttentionBars /> });
  }
  out.push({ id: "tensors", title: "Tensor stack", camera: DEFAULT_CAMERA, render: () => <TensorStack /> });
  // Model scenes lead, except when the current step has nothing to show in them.
  if (trace.raw.model.family === "cnn" && !cnnNode(trace, stepIndex) && out[0]?.id === "conv") out.push(out.shift()!);
  return out;
}

/** The honest mapping caption for a scene. */
export function sceneCaption(id: SceneId, trace: ReturnType<typeof useAppState>["trace"]): string {
  if (id === "conv") {
    return "Front planes: the input image's channels (one plane per channel, cell colour = pixel value on the diverging scale). Back planes: the four output feature maps. The translucent box is the 3×3×3 receptive-field volume (3 rows × 3 columns on every channel; padding cells lie outside the image) whose 27 products plus the bias give the selected output cell, joined to it by the dashed line. Click an output cell to move the box.";
  }
  if (id === "attention") {
    return "One bar per (row token, column token) pair for the selected sample: height = the attention weight the row token gives the column token (0 to 1, each row sums to 1; colour darkens with the weight). Row labels on the left edge, column labels on the front edge, position-numbered so repeated tokens stay apart. Click a bar or a row label to highlight that row and show its weights.";
  }
  if (id === "regions") {
    return "One box per leaf region (dashed-looking, faded boxes: nodes created but not split yet). Footprint = the region's bounds in feature space (the data range plus 0.5 on each side), height = predicted probability of class 1 at that node = count₁ / n, colour: class-0 hue at 0 to class-1 hue at 1. Floor dots: the dataset (colour = label). Highlighted box: the node being split (train) or the node the current sample is in (predict).";
  }
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
