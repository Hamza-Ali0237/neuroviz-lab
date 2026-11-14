import { StoreProvider, useAppState, type Capture } from "./state/store";
import type { TraceView } from "./trace/load";
import { TopBar } from "./layout/TopBar";
import { StepList } from "./layout/StepList";
import { Inspector } from "./layout/Inspector";
import { Playback } from "./layout/Playback";
import { CaptureFrame } from "./layout/Capture";
import { Scene2D } from "./renderers/r2d/Scene2D";
import { Scene3D } from "./renderers/r3d/Scene3D";

function Centre() {
  const { dimension } = useAppState();
  // Both renderers read and write the same store: step, selection and sample survive the switch.
  return dimension === "2d" ? <Scene2D /> : <Scene3D />;
}

function Frame() {
  const { selection, dimension } = useAppState();
  const slim = !selection.nodeId && !selection.tensorId;
  return (
    <div className={`nvz-app ${slim ? "nvz-rail" : ""} nvz-dim-${dimension}`}>
      <TopBar />
      <StepList />
      <main className="nvz-centre">
        <Centre />
      </main>
      <Inspector />
      <Playback />
    </div>
  );
}

export function App({ trace, dim = "2d", capture = null, captureSeconds = 2, scene = null }: {
  trace: TraceView;
  dim?: "2d" | "3d";
  capture?: Capture;
  captureSeconds?: number;
  scene?: string | null;
}) {
  return (
    <StoreProvider trace={trace} dim={dim} capture={capture} scene={scene}>
      {capture ? <CaptureFrame seconds={captureSeconds} /> : <Frame />}
    </StoreProvider>
  );
}
