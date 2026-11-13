import { useEffect, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { useAppState } from "../../state/store";
import { availableScenes, sceneCaption, type SceneId } from "./scenes";

/**
 * 3D renderer. Reads and writes the shared store (step, selection, sample) exactly like the
 * 2D renderer; only the camera lives here. Scenes: LR probability surface, LR loss landscape,
 * and the generic tensor stack for every other model.
 */
export function Scene3D() {
  const { trace, stepIndex } = useAppState();
  const scenes = availableScenes(trace, stepIndex);
  const [chosen, setChosen] = useState<SceneId | null>(null);
  const scene = scenes.find((s) => s.id === chosen) ?? scenes[0];
  const controls = useRef<OrbitControlsImpl>(null);
  const step = trace.steps[stepIndex];
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const on = () => setReduced(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return (
    <div className="nvz-scene3d">
      <header className="nvz-stephead">
        <span className={`nvz-phase-pill nvz-phase-${step.phase}`}>{step.phase}</span>
        <h2>{step.title}</h2>
        {trace.raw.kind === "train" && <span className="nvz-muted">iteration {step.iteration + 1}</span>}
        <span className="nvz-3d-tools">
          {scenes.length > 1 && (
            <label>
              scene
              <select value={scene.id} onChange={(e) => setChosen(e.target.value as SceneId)}>
                {scenes.map((s) => (
                  <option key={s.id} value={s.id}>{s.title}</option>
                ))}
              </select>
            </label>
          )}
          <button onClick={() => controls.current?.reset()} title="Reset camera, zoom and orbit">
            Reset view
          </button>
        </span>
      </header>
      <div className="nvz-canvas3d">
        <Canvas camera={{ position: [4.2, 3.2, 5.2], fov: 42 }} dpr={[1, 2]} gl={{ antialias: true, preserveDrawingBuffer: true }}>
          <ambientLight intensity={1.1} />
          <directionalLight position={[5, 8, 4]} intensity={1.2} />
          <directionalLight position={[-4, 3, -5]} intensity={0.4} />
          {scene.render()}
          <OrbitControls ref={controls} makeDefault enableDamping={!reduced} dampingFactor={0.15} />
        </Canvas>
      </div>
      <p className="nvz-3d-caption">
        <b>{scene.title}.</b> {sceneCaption(scene.id, trace)}
      </p>
    </div>
  );
}
