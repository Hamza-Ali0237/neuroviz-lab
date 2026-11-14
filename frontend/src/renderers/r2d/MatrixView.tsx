import { useAppState, useDispatch } from "../../state/store";
import { fmt, fmtFull, shapeStr, tint } from "../../format";
import { flatIndex } from "../../trace/load";
import { ChannelGrids } from "./ChannelGrids";
import { tokenLabels } from "../tokens";

/**
 * A tensor as a grid of numbers with a heatmap tint. 0-D: one cell; 1-D: one row;
 * N-D: leading dims flattened into rows. Click a cell to select it.
 * (N, C, H, W) tensors (and (C, H, W) in a CNN trace) are drawn as channel grids instead.
 * Rows and columns whose size is the sequence length carry token labels when the trace has them.
 */
export function MatrixView({ tensorId, caption }: { tensorId: string; caption?: string }) {
  const { trace, selection, sampleIndex } = useAppState();
  const dispatch = useDispatch();
  const t = trace.tensor(tensorId);
  const name = trace.label(tensorId);
  const nSamples = trace.raw.mode.n_samples;
  const perSample = t.role !== "param" && t.shape.length >= 1 && t.shape[0] === nSamples && nSamples > 1;

  if (!t.values) {
    return (
      <figure className="nvz-matrix">
        <figcaption>{caption ?? name} {shapeStr(t.shape)}</figcaption>
        <p className="nvz-muted">values truncated in trace · min {fmt(t.stats.min)} max {fmt(t.stats.max)}</p>
      </figure>
    );
  }
  const isImage = t.shape.length === 4 || (t.shape.length === 3 && trace.raw.model.family === "cnn");
  if (isImage) {
    const lead = t.shape.length === 4 ? t.shape[0] : 0;
    const blocks = lead ? Array.from({ length: lead }, (_, i) => [i]) : [[]];
    const isInput = t.role === "input" || t.role === "context";
    return (
      <figure className="nvz-matrix nvz-matrix-image">
        <figcaption>
          <button className="nvz-link" onClick={() => dispatch({ type: "select", selection: { tensorId } })} title={`${tensorId} · ${t.role}`}>
            {caption ?? name}
          </button>{" "}
          <span className="nvz-muted mono">{shapeStr(t.shape)}</span>
        </figcaption>
        {blocks.map((prefix) => (
          <ChannelGrids
            key={prefix.join(",")}
            tensorId={tensorId}
            prefix={prefix}
            caption={lead ? (t.role === "param" ? `${name}[${prefix[0]}]` : `sample ${prefix[0]}`) : null}
            image={isInput}
            channelLabel={t.role === "param" ? (c) => `in-channel ${c}` : lead && t.role !== "input" ? (c) => `map ${c}` : undefined}
          />
        ))}
      </figure>
    );
  }
  const cols = t.shape.length ? t.shape[t.shape.length - 1] : 1;
  const rows = t.size / cols;
  const absMax = Math.max(Math.abs(t.stats.min), Math.abs(t.stats.max), 1e-12);
  const rowIndex = (r: number): number[] => {
    const lead = t.shape.slice(0, -1);
    const idx: number[] = [];
    let rem = r;
    for (let i = lead.length - 1; i >= 0; i--) {
      idx.unshift(rem % lead[i]);
      rem = Math.floor(rem / lead[i]);
    }
    return idx;
  };
  // Token labels: a (n, T, ·) tensor labels its rows, a (·, T) tensor its columns, by the
  // tokens of the sample (the first sample for parameters such as E_pos).
  const tok = (s: number) => tokenLabels(trace, s);
  const T = tok(0)?.length ?? -1;
  const rowTokens = t.role !== "param" && t.shape.length >= 2 && t.shape[t.shape.length - 2] === T;
  const colTokens = t.role !== "param" && t.shape.length >= 2 && cols === T;
  const sampleOf = (lead: number[]) => (perSample ? lead[0] : 0);
  const selectedFlat =
    selection.tensorId === tensorId && selection.index ? flatIndex(t.shape, selection.index) : -1;
  const selectTensor = () => dispatch({ type: "select", selection: { tensorId } });
  const showRowHead = (perSample && t.shape.length > 1) || rowTokens;
  return (
    <figure className="nvz-matrix">
      <figcaption>
        <button className="nvz-link" onClick={selectTensor} title={`${tensorId} · ${t.role}`}>
          {caption ?? name}
        </button>{" "}
        <span className="nvz-muted mono">{shapeStr(t.shape)}</span>
      </figcaption>
      <table className="mono">
        {colTokens && (
          <thead>
            <tr>
              {showRowHead && <th />}
              {(tok(perSample ? sampleIndex : 0) ?? []).map((l, c) => (
                <th key={c} scope="col">{l}</th>
              ))}
            </tr>
          </thead>
        )}
        <tbody>
          {Array.from({ length: rows }, (_, r) => {
            const lead = rowIndex(r);
            const isSample = perSample && lead[0] === sampleIndex;
            const tokens = rowTokens ? tok(sampleOf(lead)) : null;
            const rowLabel = [perSample && t.shape.length > 1 ? `s${lead[0]}` : "", tokens ? tokens[lead[lead.length - 1]] : ""].filter(Boolean).join(" ");
            return (
              <tr key={r} className={isSample ? "nvz-row-sample" : ""}>
                {showRowHead && <th scope="row">{rowLabel}</th>}
                {Array.from({ length: cols }, (_, c) => {
                  const idx = t.shape.length ? [...lead, c] : [];
                  const flat = r * cols + c;
                  const v = t.values![flat];
                  const isSel = flat === selectedFlat;
                  const sampleCell = perSample && t.shape.length === 1 && idx[0] === sampleIndex;
                  return (
                    <td
                      key={c}
                      className={`nvz-cell ${isSel ? "selected" : ""} ${sampleCell ? "nvz-row-sample" : ""}`}
                      style={{ "--tint": tint(v, absMax) } as React.CSSProperties}
                      title={`${name}[${idx.join(", ")}] = ${fmtFull(v)}`}
                      onClick={() => {
                        dispatch({ type: "select", selection: { tensorId, index: idx } });
                        if (perSample) dispatch({ type: "sample", index: idx[0] });
                      }}
                    >
                      {fmt(v)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <ColourLegend absMax={absMax} />
    </figure>
  );
}

/** Diverging colour-scale legend centred on 0 (orange negative, blue positive). */
export function ColourLegend({ absMax }: { absMax: number }) {
  return (
    <div className="nvz-colour-legend mono" aria-label="Colour scale">
      <span>−{fmt(absMax)}</span>
      <i />
      <span>0</span>
      <i className="pos" />
      <span>+{fmt(absMax)}</span>
      <span className="nvz-muted">negative · 0 · positive</span>
    </div>
  );
}
