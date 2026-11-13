import { useAppState, useDispatch } from "../state/store";
import { PHASES } from "../format";
import { webglSupport } from "../renderers/r3d/webgl";

const PROVENANCE: Record<string, string> = {
  "builtin-pretrained": "Pretrained weights",
  "builtin-initial": "Initial weights",
  user: "Your weights",
};

export function TopBar() {
  const { trace, dimension } = useAppState();
  const dispatch = useDispatch();
  const { model, kind, mode, meta } = trace.raw;
  const webgl = webglSupport();
  return (
    <header className="nvz-topbar">
      <div className="nvz-brand">nvzlab</div>
      <h1 className="nvz-title">{model.name.replace(/_/g, " ")}</h1>
      <span className="nvz-badge">{kind}</span>
      <span className="nvz-badge">
        {mode.samples === "single" ? "1 sample" : `${mode.n_samples} samples`} · {mode.n_features}{" "}
        {mode.n_features === 1 ? "feature" : "features"}
      </span>
      <span className="nvz-badge nvz-badge-prov">{PROVENANCE[model.provenance] ?? model.provenance}</span>
      {kind === "train" && typeof meta.lr === "number" && (
        <span className="nvz-badge">lr = {meta.lr}</span>
      )}
      <div className="nvz-legend" aria-label="Phase legend">
        {PHASES.map((p) => (
          <span key={p} className={`nvz-legend-item nvz-phase-${p}`}>
            <i /> {p}
          </span>
        ))}
      </div>
      <div className="nvz-dim" role="group" aria-label="Renderer">
        <button
          className={dimension === "2d" ? "active" : ""}
          onClick={() => dispatch({ type: "dimension", dimension: "2d" })}
        >
          2D
        </button>
        <button
          className={dimension === "3d" ? "active" : ""}
          disabled={!webgl.ok}
          aria-disabled={!webgl.ok}
          title={webgl.ok ? "3D view (same step, selection and sample)" : `3D unavailable: ${webgl.reason}`}
          onClick={() => dispatch({ type: "dimension", dimension: "3d" })}
        >
          3D
        </button>
      </div>
    </header>
  );
}
