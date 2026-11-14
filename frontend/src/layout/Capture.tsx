import { useEffect, useState } from "react";
import { useAppState, useDispatch } from "../state/store";
import { Scene2D } from "../renderers/r2d/Scene2D";
import { Scene3D } from "../renderers/r3d/Scene3D";

export const CAPTURE_SIZES = { square: [1080, 1080], vertical: [1080, 1920] } as const;
const HOLD_MS = 1000;

/**
 * Capture mode: a fixed-size stage scaled to fit the window, no controls, a slim title
 * strip, the scene, and its caption in large type. Auto-plays every step (`seconds` each),
 * holds on the last one, then loops back to the first (`range` restricts the steps played).
 * `data-loop` counts completed loops so a recorder knows when one full pass is on tape.
 */
export function CaptureFrame({ seconds, range }: { seconds: number; range: [number, number] }) {
  const { trace, stepIndex, dimension, capture } = useAppState();
  const dispatch = useDispatch();
  const [w, h] = CAPTURE_SIZES[capture ?? "square"];
  const [scale, setScale] = useState(1);
  const [loops, setLoops] = useState(0);
  useEffect(() => {
    const fit = () => setScale(Math.min(window.innerWidth / w, window.innerHeight / h));
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, [w, h]);
  const [first, last] = range;
  useEffect(() => {
    const atEnd = stepIndex >= last;
    const id = window.setTimeout(() => {
      if (atEnd) {
        setLoops((n) => n + 1);
        dispatch({ type: "goto", step: first });
      } else dispatch({ type: "next" });
    }, atEnd ? seconds * 1000 + HOLD_MS : seconds * 1000);
    return () => window.clearTimeout(id);
  }, [stepIndex, first, last, seconds, dispatch]);
  const step = trace.steps[stepIndex];
  return (
    <div className="nvz-capture-root">
      <div
        className={`nvz-capture nvz-capture-${capture} nvz-dim-${dimension} nvz-phase-${step.phase}`}
        style={{ width: w, height: h, transform: `translate(-50%, -50%) scale(${scale})` }}
        data-step={stepIndex}
        data-loop={loops}
      >
        <header className="nvz-capture-title">
          <span className="nvz-brand">nvzlab</span>
          <span className="nvz-capture-model">{trace.raw.model.name.replace(/_/g, " ")}</span>
          <span className="nvz-capture-step">
            <i className="nvz-capture-dot" /> {step.title}
          </span>
        </header>
        <main className="nvz-capture-stage">{dimension === "2d" ? <Scene2D /> : <Scene3D />}</main>
      </div>
    </div>
  );
}
