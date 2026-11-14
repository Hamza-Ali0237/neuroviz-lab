import { useAppState, useDispatch } from "../../state/store";
import { fmt, fmtFull, tint } from "../../format";
import { flatIndex } from "../../trace/load";

/**
 * A (C, H, W) block of a tensor as a row of C small heat-map grids; `prefix` fixes the
 * leading indices (the sample, or the output channel of a kernel). With three channels and
 * `image`, the block is also drawn as an RGB picture (nearest-neighbour, values clipped to
 * [0, 1]). Cells show their number only when the grid is small (≤ 9 cells); every cell
 * has the full value on hover and selects its index on click.
 *   outline(c, h, w): cells to outline (a receptive field);
 *   windows: draw a heavier border every `windows` cells (pooling windows);
 *   winners: per channel, the flat h·W + w positions to mark (pooling argmax).
 */
export function ChannelGrids({ tensorId, prefix, caption, image, outline, windows, winners, channelLabel, channels }: {
  tensorId: string;
  prefix: number[];
  /** Draw only these channels (default: all). */
  channels?: number[];
  caption?: string | null;
  image?: boolean;
  outline?: (c: number, h: number, w: number) => boolean;
  windows?: number;
  winners?: Set<number>[];
  channelLabel?: (c: number) => string;
}) {
  const { trace, selection } = useAppState();
  const dispatch = useDispatch();
  const t = trace.tensor(tensorId);
  const name = trace.label(tensorId);
  const [C, H, W] = t.shape.slice(prefix.length);
  if (!t.values || t.shape.length - prefix.length !== 3) return null;
  const base = flatIndex(t.shape, [...prefix, 0, 0, 0]);
  const absMax = Math.max(Math.abs(t.stats.min), Math.abs(t.stats.max), 1e-12);
  const at = (c: number, h: number, w: number) => t.values![base + (c * H + h) * W + w];
  const selFlat = selection.tensorId === tensorId && selection.index ? flatIndex(t.shape, selection.index) : -1;
  const numbers = H * W <= 9; // 3×3 kernels show their numbers; larger maps are colour tiles with values on hover
  const rgb = image && C === 3;
  return (
    <figure className={`nvz-channels ${numbers ? "numbers" : ""}`}>
      {caption !== null && (
        <figcaption>
          <button className="nvz-link" onClick={() => dispatch({ type: "select", selection: { tensorId } })} title={`${tensorId} · ${t.role}`}>
            {caption ?? name}
          </button>{" "}
          {!channels && <span className="nvz-muted mono">({[C, H, W].join("×")}{prefix.length ? `, of ${t.shape.join("×")}` : ""})</span>}
        </figcaption>
      )}
      <div className="nvz-channel-row">
        {rgb && (
          <div className="nvz-channel">
            <svg className="nvz-rgb" viewBox={`0 0 ${W} ${H}`} shapeRendering="crispEdges" role="img" aria-label="RGB image">
              {Array.from({ length: H * W }, (_, k) => {
                const h = Math.floor(k / W);
                const w = k % W;
                const px = [0, 1, 2].map((c) => Math.round(Math.min(1, Math.max(0, at(c, h, w))) * 255));
                return <rect key={k} x={w} y={h} width={1} height={1} fill={`rgb(${px.join(",")})`} className={outline?.(0, h, w) ? "rf" : ""}><title>{`pixel (${h}, ${w}): R ${fmt(at(0, h, w))} G ${fmt(at(1, h, w))} B ${fmt(at(2, h, w))}`}</title></rect>;
              })}
              {outline && <OutlineRect outline={(h, w) => outline(0, h, w)} H={H} W={W} />}
            </svg>
            <span className="nvz-channel-label">as RGB</span>
          </div>
        )}
        {(channels ?? Array.from({ length: C }, (_, c) => c)).map((c) => (
          <div key={c} className="nvz-channel">
            <table className="mono" style={{ "--win": windows ?? 0 } as React.CSSProperties}>
              <tbody>
                {Array.from({ length: H }, (_, h) => (
                  <tr key={h}>
                    {Array.from({ length: W }, (_, w) => {
                      const v = at(c, h, w);
                      const idx = [...prefix, c, h, w];
                      const flat = base + (c * H + h) * W + w;
                      const cls = [
                        "nvz-cell nvz-px",
                        flat === selFlat ? "selected" : "",
                        outline?.(c, h, w) ? "rf" : "",
                        winners?.[c]?.has(h * W + w) ? "win" : "",
                        windows && h % windows === 0 ? "win-top" : "",
                        windows && w % windows === 0 ? "win-left" : "",
                      ].join(" ");
                      return (
                        <td
                          key={w}
                          className={cls}
                          style={{ "--tint": tint(v, absMax) } as React.CSSProperties}
                          title={`${name}[${idx.join(", ")}] = ${fmtFull(v)}`}
                          onClick={() => dispatch({ type: "select", selection: { tensorId, index: idx } })}
                        >
                          {numbers ? fmt(v) : ""}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <span className="nvz-channel-label">{channelLabel ? channelLabel(c) : rgb ? ["R", "G", "B"][c] : `channel ${c}`}</span>
          </div>
        ))}
      </div>
    </figure>
  );
}

/** Bounding box of the outlined cells, as one rectangle over the RGB picture. */
function OutlineRect({ outline, H, W }: { outline: (h: number, w: number) => boolean; H: number; W: number }) {
  let h0 = H, h1 = -1, w0 = W, w1 = -1;
  for (let h = 0; h < H; h++) for (let w = 0; w < W; w++) if (outline(h, w)) { h0 = Math.min(h0, h); h1 = Math.max(h1, h); w0 = Math.min(w0, w); w1 = Math.max(w1, w); }
  if (h1 < 0) return null;
  return <rect className="nvz-rf-box" x={w0} y={h0} width={w1 - w0 + 1} height={h1 - h0 + 1} />;
}
