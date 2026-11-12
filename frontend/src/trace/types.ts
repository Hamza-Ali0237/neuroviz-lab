/** TypeScript mirror of trace schema v0 (`nvzlab.trace/0`). */

export type Phase = "forward" | "loss" | "backward" | "update";
export type Role = "input" | "param" | "activation" | "output" | "grad" | "loss" | "label";

export interface Explanation {
  title: string;
  formula: string | null;
  text: string | null;
  shape_rule: string | null;
}

export interface TensorRecord {
  shape: number[];
  dtype: "float32";
  role: Role;
  name: string | null;
  stats: { min: number; max: number; mean: number };
  data: string | null;
  truncated: boolean;
}

export interface NodeRecord {
  id: string;
  op: string;
  category: string;
  origin: "builtin" | "user";
  phase: Phase;
  inputs: string[];
  outputs: string[];
  attrs: Record<string, unknown> & { grad_of?: string[] };
  step: number;
  forward_of: string | null;
  differentiable: boolean;
  explain: Explanation | null;
}

export interface EventRecord {
  id: string;
  kind: string;
  phase: Phase;
  step: number;
  payload: Record<string, unknown>;
  refs: string[];
}

export interface StepRecord {
  index: number;
  title: string;
  phase: Phase;
  iteration: number;
  nodes: string[];
  events: string[];
}

export interface ArchitectureEntry {
  id: string;
  label: string;
  op: string;
  params: string[];
}

export interface TraceRecord {
  schema: "nvzlab.trace/0";
  run_id: string;
  kind: "predict" | "train";
  model: {
    name: string;
    family: string;
    provenance: "builtin-pretrained" | "builtin-initial" | "user";
    architecture: ArchitectureEntry[];
  };
  dataset: { name: string; features: string[]; classes: string[] } | null;
  mode: { samples: "single" | "batch"; n_samples: number; n_features: number };
  tensors: Record<string, TensorRecord>;
  nodes: NodeRecord[];
  events: EventRecord[];
  steps: StepRecord[];
  meta: Record<string, unknown> & { nvzlab_version?: string; lr?: number | null };
}
