import { useMemo } from "react";
import * as THREE from "three";
import { Html } from "@react-three/drei";
import { Line } from "@react-three/drei";
import { NEG_RGB, POS_RGB, fmt, ticks, tintAlpha } from "../../format";

/** Side length of the cube every scene is drawn into (axes map their range onto it). */
export const SIZE = 4;

/** A CSS colour from the page theme (light/dark aware), as a THREE.Color. */
export function themeColor(name: string): THREE.Color {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return new THREE.Color(v || "#888888");
}

export function useTheme() {
  // Re-read on every mount of the 3D view, so a theme switch is picked up on the next toggle.
  return useMemo(
    () => ({
      fg: themeColor("--fg"),
      muted: themeColor("--muted"),
      panel: themeColor("--panel"),
      border: themeColor("--border"),
      accent: themeColor("--accent"),
      pt0: themeColor("--pt0"),
      pt1: themeColor("--pt1"),
      sel: themeColor("--sel"),
    }),
    [],
  );
}

/** Same diverging tint as the 2D grids: blue positive, orange negative, blended on the panel colour. */
export function cellColor(v: number, absMax: number, panel: THREE.Color): THREE.Color {
  const [r, g, b] = v >= 0 ? POS_RGB : NEG_RGB;
  const a = tintAlpha(v, absMax);
  return new THREE.Color(r / 255, g / 255, b / 255).lerp(panel, 1 - a);
}

/** Camera position at `elevation`/`azimuth` degrees and `distance` from the origin. */
export function orbitPosition(elevation: number, azimuth: number, distance: number): [number, number, number] {
  const el = (elevation * Math.PI) / 180;
  const az = (azimuth * Math.PI) / 180;
  return [distance * Math.cos(el) * Math.sin(az), distance * Math.sin(el), distance * Math.cos(el) * Math.cos(az)];
}

/** Map a value in [lo, hi] onto [-SIZE/2, SIZE/2]. */
export function span(lo: number, hi: number) {
  return (v: number) => ((v - lo) / (hi - lo || 1) - 0.5) * SIZE;
}

/**
 * A height-field surface from a grid `z[i][j]` (i along x, j along depth), with vertex
 * colours from `color(value)`. Heights are already mapped to scene units by the caller.
 */
export function Surface({ xs, zs, heights, color, opacity = 1 }: {
  xs: number[];
  zs: number[];
  heights: (i: number, j: number) => number;
  color: (i: number, j: number) => THREE.Color;
  opacity?: number;
}) {
  const geometry = useMemo(() => {
    const nx = xs.length;
    const nz = zs.length;
    const pos = new Float32Array(nx * nz * 3);
    const col = new Float32Array(nx * nz * 3);
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < nz; j++) {
        const k = (i * nz + j) * 3;
        pos[k] = xs[i];
        pos[k + 1] = heights(i, j);
        pos[k + 2] = zs[j];
        const c = color(i, j);
        col[k] = c.r;
        col[k + 1] = c.g;
        col[k + 2] = c.b;
      }
    }
    const idx: number[] = [];
    for (let i = 0; i < nx - 1; i++) {
      for (let j = 0; j < nz - 1; j++) {
        const a = i * nz + j;
        const b = a + nz;
        idx.push(a, a + 1, b, b, a + 1, b + 1);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
  }, [xs, zs, heights, color]);
  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial vertexColors side={THREE.DoubleSide} transparent={opacity < 1} opacity={opacity} roughness={0.9} />
    </mesh>
  );
}

/** A DOM label anchored at a 3D point. */
export function Label({ at, children, muted, onClick, title }: {
  at: [number, number, number];
  children: React.ReactNode;
  muted?: boolean;
  onClick?: () => void;
  title?: string;
}) {
  return (
    <Html position={at} center className={`nvz-3d-label ${muted ? "muted" : ""} ${onClick ? "clickable" : ""}`} style={{ pointerEvents: onClick ? "auto" : "none" }} zIndexRange={[10, 0]}>
      <span onClick={onClick} title={title}>{children}</span>
    </Html>
  );
}

/**
 * Floor grid and the three axes drawn on the cube's edges: x along the front floor edge,
 * z along the right floor edge, y up the back-left corner. Ticks are "nice" values inside
 * each range; titles sit just outside the edge, never in mid-air.
 */
export function Axes({ x, z, y, colors, yHeight = SIZE / 2, ticks: showTicks = true }: {
  x: { name: string; lo: number; hi: number };
  z: { name: string; lo: number; hi: number };
  y: { name: string; lo: number; hi: number };
  colors: { border: THREE.Color; muted: THREE.Color };
  /** Scene height that y.hi maps to (default half the cube). */
  yHeight?: number;
  /** Draw tick marks and tick labels (off for categorical axes). */
  ticks?: boolean;
}) {
  const h = SIZE / 2;
  const eps = 1e-9;
  const inside = (a: { lo: number; hi: number }, n: number) => (showTicks ? ticks(a.lo, a.hi, n).filter((v) => v >= a.lo - eps && v <= a.hi + eps) : []);
  const sx = span(x.lo, x.hi);
  const sz = span(z.lo, z.hi);
  const sy = (v: number) => ((v - y.lo) / (y.hi - y.lo || 1)) * yHeight;
  const tick = 0.12;
  const c = colors.border;
  return (
    <group>
      <gridHelper args={[SIZE, 8, c, c]} position={[0, 0, 0]} />
      <Line points={[[-h, 0, h], [h, 0, h]]} color={c} lineWidth={1.5} />
      <Line points={[[h, 0, -h], [h, 0, h]]} color={c} lineWidth={1.5} />
      <Line points={[[-h, 0, -h], [-h, yHeight, -h]]} color={c} lineWidth={1.5} />
      {inside(x, 4).map((v) => (
        <group key={`x${v}`}>
          <Line points={[[sx(v), 0, h], [sx(v), 0, h + tick]]} color={c} lineWidth={1} />
          <Label at={[sx(v), 0, h + 0.42]} muted>{fmt(v)}</Label>
        </group>
      ))}
      <Label at={[0, 0, h + 0.95]}>{x.name}</Label>
      {inside(z, 4).map((v) => (
        <group key={`z${v}`}>
          <Line points={[[h, 0, sz(v)], [h + tick, 0, sz(v)]]} color={c} lineWidth={1} />
          <Label at={[h + 0.5, 0, sz(v)]} muted>{fmt(v)}</Label>
        </group>
      ))}
      <Label at={[h + 1.05, 0, 0]}>{z.name}</Label>
      {inside(y, 3).map((v) => (
        <group key={`y${v}`}>
          <Line points={[[-h, sy(v), -h], [-h - tick, sy(v), -h]]} color={c} lineWidth={1} />
          <Label at={[-h - 0.45, sy(v), -h]} muted>{fmt(v)}</Label>
        </group>
      ))}
      <Label at={[-h - 0.45, yHeight + 0.3, -h]}>{y.name}</Label>
    </group>
  );
}

/** Bilinear interpolation of grid[i][j] at (a, b) on axes ax0 (i) and ax1 (j); null outside. */
export function bilinear(grid: Float32Array, ax0: Float32Array, ax1: Float32Array, a: number, b: number): number | null {
  const n0 = ax0.length;
  const n1 = ax1.length;
  const fi = ((a - ax0[0]) / (ax0[n0 - 1] - ax0[0])) * (n0 - 1);
  const fj = ((b - ax1[0]) / (ax1[n1 - 1] - ax1[0])) * (n1 - 1);
  if (fi < 0 || fj < 0 || fi > n0 - 1 || fj > n1 - 1) return null;
  const i = Math.min(Math.floor(fi), n0 - 2);
  const j = Math.min(Math.floor(fj), n1 - 2);
  const u = fi - i;
  const v = fj - j;
  const g = (ii: number, jj: number) => grid[ii * n1 + jj];
  return (1 - u) * (1 - v) * g(i, j) + u * (1 - v) * g(i + 1, j) + (1 - u) * v * g(i, j + 1) + u * v * g(i + 1, j + 1);
}
