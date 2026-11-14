import type { TraceView } from "../trace/load";
import type { EventRecord } from "../trace/types";

export interface TreeNode {
  id: number;
  depth: number;
  parent: number | null;
  side: "left" | "right" | null;
  feature: number | null;
  feature_name: string | null;
  threshold: number | null;
  left: number | null;
  right: number | null;
  counts: number[];
  n_samples: number;
  impurity: number | null;
  /** Declared a leaf (predict: by the structure; train: by a `leaf` event). */
  is_leaf: boolean;
  /** Train: created but not decided yet at this step. */
  pending: boolean;
  predicted_class: number;
  probability: number;
  reason: string | null;
}

export interface SplitCandidate {
  feature: number;
  feature_name: string;
  threshold: number;
  left_counts: number[];
  right_counts: number[];
  left_impurity: number;
  right_impurity: number;
  weighted_impurity: number;
  gain: number;
}

export interface SplitSearchView {
  node: number;
  depth: number;
  n_samples: number;
  counts: number[];
  impurity: number;
  criterion: string;
  explain: { title: string; formula: string; text: string };
  candidates: SplitCandidate[];
  chosen: (SplitCandidate & { candidate: number }) | null;
}

export interface TreeState {
  nodes: TreeNode[];
  features: string[];
  criterion: string | null;
  /** Predict: node ids visited by each sample up to this step (root first). */
  paths: number[][];
  /** Train: the split search recorded in this step, if any. */
  split: SplitSearchView | null;
  /** Node ids created in this step (train). */
  created: number[];
}

const majority = (counts: number[]) => {
  const total = counts.reduce((a, b) => a + b, 0);
  const pc = counts.indexOf(Math.max(...counts));
  return { predicted_class: pc, probability: total ? counts[pc] / total : 0 };
};

function fromStructure(ev: EventRecord): { nodes: TreeNode[]; features: string[]; criterion: string } {
  const p = ev.payload as { criterion: string; features: string[]; nodes: Record<string, unknown>[] };
  const nodes = p.nodes.map((n) => {
    const node: TreeNode = {
      id: n.id as number,
      depth: n.depth as number,
      parent: null,
      side: null,
      feature: (n.feature as number | null) ?? null,
      feature_name: (n.feature_name as string | null) ?? null,
      threshold: (n.threshold as number | null) ?? null,
      left: (n.left as number | null) ?? null,
      right: (n.right as number | null) ?? null,
      counts: n.counts as number[],
      n_samples: n.n_samples as number,
      impurity: n.impurity as number,
      is_leaf: n.is_leaf as boolean,
      pending: false,
      predicted_class: n.predicted_class as number,
      probability: n.probability as number,
      reason: null,
    };
    return node;
  });
  for (const n of nodes) {
    for (const [side, child] of [["left", n.left], ["right", n.right]] as const) {
      if (child !== null) {
        nodes[child].parent = n.id;
        nodes[child].side = side;
      }
    }
  }
  return { nodes, features: p.features, criterion: p.criterion };
}

/**
 * The tree as known at a step. Predict traces: the structure from the `tree_structure`
 * event plus the path each sample has taken so far. Train traces: the tree rebuilt from the
 * growth events up to this step (nodes created, splits chosen, leaves declared).
 */
export function treeAt(trace: TraceView, stepIndex: number): TreeState | null {
  const events = trace.raw.events.filter((e) => e.step <= stepIndex);
  const features = trace.raw.dataset?.features ?? [];
  const structure = events.find((e) => e.kind === "tree_structure");
  const n = trace.raw.mode.n_samples;
  if (trace.raw.kind === "predict") {
    if (!structure) return null;
    const { nodes, features: f, criterion } = fromStructure(structure);
    const paths = Array.from({ length: n }, () => [0]);
    for (const e of events) {
      if (e.kind === "node_test") paths[e.payload.sample as number].push(e.payload.next_node as number);
    }
    return { nodes, features: f, criterion, paths, split: null, created: [] };
  }
  const nodes: TreeNode[] = [];
  const ensure = (id: number, depth: number, counts: number[], impurity: number | null, parent: number | null, side: "left" | "right" | null): TreeNode => {
    if (nodes[id]) return nodes[id];
    const total = counts.reduce((a, b) => a + b, 0);
    nodes[id] = { id, depth, parent, side, feature: null, feature_name: null, threshold: null, left: null, right: null, counts, n_samples: total, impurity, is_leaf: false, pending: true, reason: null, ...majority(counts) };
    return nodes[id];
  };
  let split: SplitSearchView | null = null;
  const created: number[] = [];
  let criterion: string | null = null;
  for (const e of events) {
    const p = e.payload as Record<string, unknown>;
    const here = e.step === stepIndex;
    if (e.kind === "split_search") {
      const node = ensure(p.node as number, p.depth as number, p.counts as number[], p.impurity as number, null, null);
      criterion = p.criterion as string;
      if (here) split = { ...(p as unknown as SplitSearchView), chosen: null, node: node.id };
    } else if (e.kind === "split_chosen") {
      const node = nodes[p.node as number];
      if (node) {
        node.feature = p.feature as number;
        node.feature_name = p.feature_name as string;
        node.threshold = p.threshold as number;
        node.pending = false;
      }
      if (here && split && split.node === p.node) split.chosen = p as unknown as SplitSearchView["chosen"];
    } else if (e.kind === "node_created") {
      const parent = nodes[p.parent as number];
      const child = ensure(p.id as number, p.depth as number, p.counts as number[], p.impurity as number, p.parent as number, p.side as "left" | "right");
      if (parent) parent[p.side as "left" | "right"] = child.id;
      if (here) created.push(child.id);
    } else if (e.kind === "leaf" && typeof p.node === "number") {
      const node = nodes[p.node] ?? ensure(p.node, 0, p.counts as number[], null, null, null);
      node.is_leaf = true;
      node.pending = false;
      node.reason = (p.reason as string | undefined) ?? null;
      Object.assign(node, majority(node.counts));
    }
  }
  if (!nodes.length) return null;
  const list = nodes.filter(Boolean);
  if (structure) {
    const full = fromStructure(structure);
    return { ...full, paths: [], split, created };
  }
  return { nodes: list, features, criterion, paths: [], split, created };
}

export interface Region {
  node: TreeNode;
  /** Bounds per feature: [lo, hi]. */
  bounds: [number, number][];
}

/** Leaf (or not-yet-split) regions of feature space inside `bounds`, by walking the splits. */
export function regions(tree: TreeState, bounds: [number, number][]): Region[] {
  const out: Region[] = [];
  const walk = (id: number, b: [number, number][]) => {
    const node = tree.nodes.find((nd) => nd.id === id);
    if (!node) return;
    if (node.feature === null || node.threshold === null || node.left === null || node.right === null) {
      out.push({ node, bounds: b });
      return;
    }
    const f = node.feature;
    const l = b.map((r) => [...r] as [number, number]);
    const r = b.map((rr) => [...rr] as [number, number]);
    l[f][1] = Math.min(l[f][1], node.threshold);
    r[f][0] = Math.max(r[f][0], node.threshold);
    walk(node.left, l);
    walk(node.right, r);
  };
  walk(0, bounds);
  return out;
}

/** Feature-space bounds: the data range plus a margin. */
export function dataBounds(trace: TraceView, margin = 0.5): [number, number][] | null {
  const ctx = trace.context("dataset_X") ?? trace.byRole("input", "x");
  if (!ctx?.values || ctx.shape.length !== 2) return null;
  const nf = ctx.shape[1];
  return Array.from({ length: nf }, (_, j) => {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < ctx.shape[0]; i++) {
      const v = ctx.values![i * nf + j];
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    return [lo - margin, hi + margin];
  });
}

/** "x1 ≤ −0.47" for a split node. */
export function rule(node: TreeNode, digits = 3): string {
  if (node.feature_name === null || node.threshold === null) return "";
  return `${node.feature_name} ≤ ${Number(node.threshold.toPrecision(digits)).toString().replace("-", "−")}`;
}
