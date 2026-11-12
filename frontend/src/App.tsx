import { StoreProvider, useAppState } from "./state/store";
import type { TraceView } from "./trace/load";
import { TopBar } from "./layout/TopBar";
import { StepList } from "./layout/StepList";
import { Inspector } from "./layout/Inspector";
import { Playback } from "./layout/Playback";
import { Scene2D } from "./renderers/r2d/Scene2D";

function Centre() {
  const { dimension } = useAppState();
  // The 3D renderer plugs in here; it reads the same store and trace.
  return dimension === "2d" ? <Scene2D /> : <div className="nvz-empty">3D view coming next.</div>;
}

export function App({ trace }: { trace: TraceView }) {
  return (
    <StoreProvider trace={trace}>
      <div className="nvz-app">
        <TopBar />
        <StepList />
        <main className="nvz-centre">
          <Centre />
        </main>
        <Inspector />
        <Playback />
      </div>
    </StoreProvider>
  );
}
