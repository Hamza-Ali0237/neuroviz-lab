import { useEffect, useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import type { PerspectiveCamera } from "three";
import { OrbitControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { useAppState } from "../../state/store";
import { availableScenes, sceneCaption, type SceneId } from "./scenes";

/** Half-extents (scene units) that must stay in frame: the SIZE cube's diagonal plus label room. */
const FRAME_HALF_W = 3.3;
const FRAME_HALF_H = 3.0;
/** Scenes sit on a floor at y = -SIZE/4, so the camera looks a little below the origin. */
const LOOK_AT_Y = -0.6;

/**
 * Puts the camera on the `direction` ray, looking at the origin, at the distance that fits
 * FRAME_HALF_W × FRAME_HALF_H for the canvas aspect; re-runs when the scene or aspect
 * changes and the orbit controls reset there. The camera never moves on its own.
 */
function CameraRig({ direction }: { direction: [number, number, number] }) {
  const { camera, controls, size } = useThree();
  const aspect = size.width / Math.max(1, size.height);
  const fov = ((camera as PerspectiveCamera).fov ?? 42) * (Math.PI / 180);
  const tv = Math.tan(fov / 2);
  const distance = Math.max(FRAME_HALF_H / tv, FRAME_HALF_W / (tv * aspect));
  const key = `${direction.join(",")}:${distance.toFixed(3)}`;
  useEffect(() => {
    const n = Math.hypot(...direction) || 1;
    camera.position.set((direction[0] / n) * distance, (direction[1] / n) * distance, (direction[2] / n) * distance);
    camera.lookAt(0, LOOK_AT_Y, 0);
    camera.updateProjectionMatrix();
    const c = controls as OrbitControlsImpl | null;
    c?.target.set(0, LOOK_AT_Y, 0);
    c?.saveState?.();
  }, [camera, controls, key]); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}

/**
 * 3D renderer. Reads and writes the shared store (step, selection, sample) exactly like the
 * 2D renderer; only the camera lives here. Scenes: LR probability surface, LR loss landscape,
 * MLP network and surface, tree regions, and the generic tensor stack for every other model.
 * In capture mode the header and orbit controls are hidden and the camera is static.
 */
export function Scene3D() {
  const { trace, stepIndex, capture, scene: preferred } = useAppState();
  const scenes = availableScenes(trace, stepIndex);
  const [chosen, setChosen] = useState<SceneId | null>(preferred as SceneId | null);
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
  // Colour scheme: remount the canvas when it flips so every material re-reads the theme.
  const [dark, setDark] = useState(() => window.matchMedia("(prefers-color-scheme: dark)").matches);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const on = () => requestAnimationFrame(() => setDark(mq.matches)); // after the new CSS variables apply
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return (
    <div className="nvz-scene3d">
      {!capture && (
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
      )}
      <div className="nvz-canvas3d">
        <Canvas key={dark ? "dark" : "light"} camera={{ position: scene.camera, fov: 42 }} dpr={[1, 2]} resize={{ offsetSize: true }} gl={{ antialias: true, preserveDrawingBuffer: true }}>
          <ambientLight intensity={1.1} />
          <directionalLight position={[5, 8, 4]} intensity={1.2} />
          <directionalLight position={[-4, 3, -5]} intensity={0.4} />
          {scene.render()}
          <CameraRig direction={scene.camera} />
          {!capture && <OrbitControls ref={controls} makeDefault enableDamping={!reduced} dampingFactor={0.15} />}
        </Canvas>
      </div>
      <p className="nvz-3d-caption">
        <b>{scene.title}.</b> {sceneCaption(scene.id, trace)}
      </p>
    </div>
  );
}
