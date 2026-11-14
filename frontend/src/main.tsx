import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "katex/dist/katex.min.css";
import "./styles.css";
import { App } from "./App";
import { loadTrace } from "./trace/load";
import type { TraceRecord } from "./trace/types";
import type { Capture } from "./state/store";

async function readTrace(): Promise<TraceRecord> {
  // Exported HTML embeds the trace in a JSON script tag; `npm run dev` fetches the sample.
  const embedded = document.getElementById("nvz-trace");
  if (embedded?.textContent) return JSON.parse(embedded.textContent) as TraceRecord;
  const res = await fetch("/dev/sample-trace.json");
  if (!res.ok) throw new Error(`Could not load dev/sample-trace.json (${res.status})`);
  return (await res.json()) as TraceRecord;
}

// Page flags: data attributes written by Python, overridable from the URL
// (?dim=3d&capture=square&seconds=2&scene=loss) so a recording needs no re-export.
const rootEl = document.getElementById("root")!;
const query = new URLSearchParams(window.location.search);
const flag = (name: string, attr: string) => query.get(name) || rootEl.dataset[attr] || "";
const dim = flag("dim", "dim") === "3d" ? "3d" : "2d";
const captureFlag = flag("capture", "capture");
const capture: Capture = captureFlag === "square" || captureFlag === "vertical" ? captureFlag : null;
const seconds = Number(flag("seconds", "captureSeconds")) || 2;
const scene = flag("scene", "scene") || null;

const root = createRoot(rootEl);
readTrace()
  .then((raw) => {
    root.render(
      <StrictMode>
        <App trace={loadTrace(raw)} dim={dim} capture={capture} captureSeconds={seconds} scene={scene} />
      </StrictMode>,
    );
  })
  .catch((err: unknown) => {
    root.render(<pre className="nvz-fatal">Could not load the trace: {String(err)}</pre>);
  });
