import type { TraceView } from "../trace/load";
import { sub } from "../format";

/**
 * Token labels per sample ("C₁ B₂ E₃ …", position-numbered so repeats stay distinguishable),
 * read from the Input event payload (`tokens`, `vocab`). Null when the trace has none.
 */
export function tokenLabels(trace: TraceView, sample: number): string[] | null {
  const ev = trace.raw.events.find((e) => e.kind === "input" && Array.isArray(e.payload.tokens));
  const rows = ev?.payload.tokens as string[][] | undefined;
  const row = rows?.[sample];
  return row ? row.map((t, i) => `${t}${sub(i + 1)}`) : null;
}

/** Sequence length T from the Input event, or null. */
export function seqLen(trace: TraceView): number | null {
  const ev = trace.raw.events.find((e) => e.kind === "input" && typeof e.payload.seq_len === "number");
  return ev ? (ev.payload.seq_len as number) : null;
}

/** Vocabulary from the Input event, or null. */
export function vocabOf(trace: TraceView): string[] | null {
  const ev = trace.raw.events.find((e) => e.kind === "input" && Array.isArray(e.payload.vocab));
  return ev ? (ev.payload.vocab as string[]) : null;
}
