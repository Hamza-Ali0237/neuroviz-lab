import { createContext, useContext, useEffect, useReducer, type Dispatch, type ReactNode } from "react";
import type { TraceView } from "../trace/load";

export interface Selection {
  nodeId: string | null;
  tensorId: string | null;
  index: number[] | null;
}

export interface AppState {
  trace: TraceView;
  stepIndex: number;
  playing: boolean;
  /** Playback speed multiplier (0.5–2). */
  speed: number;
  selection: Selection;
  sampleIndex: number;
  dimension: "2d" | "3d";
}

export type Action =
  | { type: "next" }
  | { type: "prev" }
  | { type: "goto"; step: number }
  | { type: "play" }
  | { type: "pause" }
  | { type: "toggle" }
  | { type: "speed"; speed: number }
  | { type: "select"; selection: Partial<Selection> }
  | { type: "clearSelection" }
  | { type: "sample"; index: number }
  | { type: "dimension"; dimension: "2d" | "3d" };

const EMPTY: Selection = { nodeId: null, tensorId: null, index: null };

export function reducer(s: AppState, a: Action): AppState {
  const last = s.trace.steps.length - 1;
  const clamp = (i: number) => Math.max(0, Math.min(last, i));
  switch (a.type) {
    case "next":
      return s.stepIndex >= last
        ? { ...s, playing: false }
        : { ...s, stepIndex: s.stepIndex + 1, selection: EMPTY };
    case "prev":
      return { ...s, stepIndex: clamp(s.stepIndex - 1), selection: EMPTY };
    case "goto":
      return { ...s, stepIndex: clamp(a.step), selection: EMPTY };
    case "play":
      return { ...s, playing: true, stepIndex: s.stepIndex >= last ? 0 : s.stepIndex };
    case "pause":
      return { ...s, playing: false };
    case "toggle":
      return reducer(s, { type: s.playing ? "pause" : "play" });
    case "speed":
      return { ...s, speed: Math.max(0.5, Math.min(2, a.speed)) };
    case "select":
      return { ...s, selection: { ...EMPTY, ...a.selection } };
    case "clearSelection":
      return { ...s, selection: EMPTY };
    case "sample":
      return { ...s, sampleIndex: Math.max(0, Math.min(s.trace.raw.mode.n_samples - 1, a.index)) };
    case "dimension":
      return { ...s, dimension: a.dimension };
  }
}

const StateCtx = createContext<AppState | null>(null);
const DispatchCtx = createContext<Dispatch<Action> | null>(null);

export function StoreProvider({ trace, children }: { trace: TraceView; children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, {
    trace,
    stepIndex: 0,
    playing: false,
    speed: 1,
    selection: EMPTY,
    sampleIndex: 0,
    dimension: "2d",
  });

  // Playback timer: one step per second at 1×.
  useEffect(() => {
    if (!state.playing) return;
    const id = window.setInterval(() => dispatch({ type: "next" }), 1000 / state.speed);
    return () => window.clearInterval(id);
  }, [state.playing, state.speed]);

  // Keyboard: ← / → step, space play/pause.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.key === "ArrowRight") dispatch({ type: "next" });
      else if (e.key === "ArrowLeft") dispatch({ type: "prev" });
      else if (e.key === " ") dispatch({ type: "toggle" });
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <StateCtx.Provider value={state}>
      <DispatchCtx.Provider value={dispatch}>{children}</DispatchCtx.Provider>
    </StateCtx.Provider>
  );
}

export function useAppState(): AppState {
  const s = useContext(StateCtx);
  if (!s) throw new Error("useAppState outside StoreProvider");
  return s;
}

export function useDispatch(): Dispatch<Action> {
  const d = useContext(DispatchCtx);
  if (!d) throw new Error("useDispatch outside StoreProvider");
  return d;
}
