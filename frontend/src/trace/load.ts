import type { EventRecord, NodeRecord, TensorRecord, TraceRecord } from "./types";

/** Base64 little-endian float32 → Float32Array. */
export function decodeFloat32(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  const out = new Float32Array(bytes.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

export interface Tensor extends TensorRecord {
  id: string;
  /** Decoded values (row-major), or null when truncated. */
  values: Float32Array | null;
  size: number;
}

/** Row-major flat offset of a multi-index. */
export function flatIndex(shape: number[], index: number[]): number {
  let off = 0;
  for (let i = 0; i < shape.length; i++) off = off * shape[i] + (index[i] ?? 0);
  return off;
}

/** Inverse of flatIndex. */
export function unflatIndex(shape: number[], flat: number): number[] {
  const idx = new Array<number>(shape.length);
  for (let i = shape.length - 1; i >= 0; i--) {
    idx[i] = flat % shape[i];
    flat = Math.floor(flat / shape[i]);
  }
  return idx;
}

/** A loaded trace: the raw record plus decoded tensors and lookup helpers. */
export class TraceView {
  readonly raw: TraceRecord;
  readonly tensors = new Map<string, Tensor>();
  private readonly nodeById = new Map<string, NodeRecord>();
  private readonly eventById = new Map<string, EventRecord>();
  private readonly producerOf = new Map<string, NodeRecord>();

  constructor(raw: TraceRecord) {
    if (raw.schema !== "nvzlab.trace/0") {
      throw new Error(`Unsupported trace schema ${String(raw.schema)}; expected nvzlab.trace/0`);
    }
    this.raw = raw;
    for (const [id, t] of Object.entries(raw.tensors)) {
      const size = t.shape.reduce((a, b) => a * b, 1);
      this.tensors.set(id, { ...t, id, size, values: t.data ? decodeFloat32(t.data) : null });
    }
    for (const n of raw.nodes) {
      this.nodeById.set(n.id, n);
      for (const o of n.outputs) this.producerOf.set(o, n);
    }
    for (const e of raw.events) this.eventById.set(e.id, e);
  }

  get steps() {
    return this.raw.steps;
  }

  tensor(id: string): Tensor {
    const t = this.tensors.get(id);
    if (!t) throw new Error(`Unknown tensor ${id}`);
    return t;
  }

  node(id: string): NodeRecord {
    const n = this.nodeById.get(id);
    if (!n) throw new Error(`Unknown node ${id}`);
    return n;
  }

  event(id: string): EventRecord {
    const e = this.eventById.get(id);
    if (!e) throw new Error(`Unknown event ${id}`);
    return e;
  }

  nodesInStep(i: number): NodeRecord[] {
    return (this.raw.steps[i]?.nodes ?? []).map((id) => this.node(id));
  }

  eventsInStep(i: number): EventRecord[] {
    return (this.raw.steps[i]?.events ?? []).map((id) => this.event(id));
  }

  /** The node that produced a tensor, or null for leaves (inputs, params, labels). */
  producer(tensorId: string): NodeRecord | null {
    return this.producerOf.get(tensorId) ?? null;
  }

  /** Scalar at a multi-index; null if the tensor data was truncated. */
  value(tensorId: string, index: number[]): number | null {
    const t = this.tensor(tensorId);
    if (!t.values) return null;
    return t.values[flatIndex(t.shape, index)];
  }

  /** Display name for a tensor: its name, or the op that produced it. */
  label(tensorId: string): string {
    const t = this.tensor(tensorId);
    if (t.name) return t.name;
    const p = this.producer(tensorId);
    return p ? `${p.op} out` : tensorId;
  }

  /**
   * Parameter tensors visible at a step: the first-seen params, replaced by the `after`
   * tensor of every sgd_update event at or before the step.
   */
  paramsAt(stepIndex: number): Map<string, Tensor> {
    const out = new Map<string, Tensor>();
    for (const t of this.tensors.values()) {
      if (t.role === "param" && t.name && !this.producerOf.has(t.id) && !out.has(t.name)) {
        if (!this.raw.events.some((e) => e.kind === "sgd_update" && e.payload.after === t.id)) {
          out.set(t.name, t);
        }
      }
    }
    for (const e of this.raw.events) {
      if (e.kind === "sgd_update" && e.step <= stepIndex) {
        out.set(String(e.payload.param), this.tensor(String(e.payload.after)));
      }
    }
    return out;
  }
}

export function loadTrace(raw: TraceRecord): TraceView {
  return new TraceView(raw);
}
