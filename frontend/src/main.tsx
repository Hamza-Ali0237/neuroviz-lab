import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "katex/dist/katex.min.css";
import "./styles.css";
import { App } from "./App";
import { loadTrace } from "./trace/load";
import type { TraceRecord } from "./trace/types";

async function readTrace(): Promise<TraceRecord> {
  // Exported HTML embeds the trace in a JSON script tag; `npm run dev` fetches the sample.
  const embedded = document.getElementById("nvz-trace");
  if (embedded?.textContent) return JSON.parse(embedded.textContent) as TraceRecord;
  const res = await fetch("/dev/sample-trace.json");
  if (!res.ok) throw new Error(`Could not load dev/sample-trace.json (${res.status})`);
  return (await res.json()) as TraceRecord;
}

const root = createRoot(document.getElementById("root")!);
readTrace()
  .then((raw) => {
    root.render(
      <StrictMode>
        <App trace={loadTrace(raw)} />
      </StrictMode>,
    );
  })
  .catch((err: unknown) => {
    root.render(<pre className="nvz-fatal">Could not load the trace: {String(err)}</pre>);
  });
