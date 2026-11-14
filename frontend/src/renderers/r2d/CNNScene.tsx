import { useAppState } from "../../state/store";
import type { NodeRecord } from "../../trace/types";
import { fmt, sub } from "../../format";
import { ChannelGrids } from "./ChannelGrids";
import { Worked } from "./WorkedView";

/** The conv2d or maxpool2d node of a step (forward or backward), or null. */
export function cnnNode(trace: ReturnType<typeof useAppState>["trace"], stepIndex: number): NodeRecord | null {
  return trace.nodesInStep(stepIndex).find((n) => n.op === "conv2d" || n.op === "maxpool2d") ?? null;
}

/** Pool winners per channel for a sample: the flat h·W + w positions from the forward node's argmax. */
export function poolWinners(fwd: NodeRecord, sample: number): Set<number>[] {
  const argmax = fwd.attrs.argmax as number[][][][];
  return (argmax[sample] ?? []).map((rows) => new Set(rows.flat()));
}

/**
 * CNN scenes. Convolution: the input with the selected output cell's 3×3 receptive field
 * outlined on every channel, the kernel that produced it, the output maps, and the 27-term
 * worked sum. Max pool: the 2×2 windows with their winners; on the backward step the routed
 * gradient (only the winners receive it). Other steps show nothing special here.
 */
export function CNNScene() {
  const { trace, stepIndex, selection, sampleIndex, capture } = useAppState();
  const node = cnnNode(trace, stepIndex);
  if (!node) return null;
  const backward = node.phase === "backward";
  const fwd = backward ? trace.node(node.forward_of!) : node;
  const outId = node.outputs[0];
  const out = trace.tensor(outId);
  const nS = out.shape[0];
  const sample = Math.min(sampleIndex, nS - 1);
  // The selected output cell (forward), or the selected upstream-gradient cell (backward).
  const keyId = backward ? node.inputs[0] : outId;
  const selHere = selection.tensorId === keyId && selection.index?.length === 4 ? selection.index : null;
  const idx = selHere ?? [sample, 0, 0, 0];
  const [n, o, i, j] = idx;

  if (node.op === "conv2d" && !backward) {
    const [xId, wId] = node.inputs;
    const [, , kh, kw] = trace.tensor(wId).shape;
    const s = (node.attrs.stride as number | undefined) ?? 1;
    const p = (node.attrs.padding as number | undefined) ?? 0;
    const inField = (h: number, w: number) => h >= i * s - p && h < i * s - p + kh && w >= j * s - p && w < j * s - p + kw;
    return (
      <figure className="nvz-scene nvz-cnn">
        <figcaption>
          Convolution: output map {o}, cell ({i}, {j}) <span className="nvz-muted">= the 3×3 window on every input channel · kernel {o} + bias.{capture ? "" : " Click any output cell to move the window."}</span>
        </figcaption>
        <div className="nvz-cnn-row">
          <ChannelGrids tensorId={xId} prefix={[n]} caption={`input x, sample ${n}`} image outline={(_, h, w) => inField(h, w)} />
          <span className="nvz-cnn-op">⊛</span>
          <ChannelGrids tensorId={wId} prefix={[o]} caption={`kernel ${o} (W_conv[${o}])`} channelLabel={(c) => `× channel ${c}`} />
          <span className="nvz-cnn-op">=</span>
          <ChannelGrids tensorId={outId} prefix={[n]} caption={capture ? `output map ${o}` : `output z, sample ${n}`} channelLabel={(c) => `map ${c}`} outline={(c, h, w) => c === o && h === i && w === j} channels={capture ? [o] : undefined} />
        </div>
        <Worked node={node} outId={outId} index={idx} />
      </figure>
    );
  }
  if (node.op === "maxpool2d") {
    const size = (fwd.attrs.size as number | undefined) ?? 2;
    const winners = poolWinners(fwd, n);
    const inId = backward ? fwd.inputs[0] : node.inputs[0];
    const grads = backward ? outId : null;
    const inShape = trace.tensor(inId).shape;
    const nIn = (grads ? trace.tensor(grads) : trace.tensor(inId)).values;
    const zeros = grads && nIn ? Array.from(nIn).filter((v) => v === 0).length : 0;
    // Backward: the selected gradient cell, else the winner of the window at the selected upstream cell.
    const W = inShape[3];
    const win = (fwd.attrs.argmax as number[][][][])[n]?.[o]?.[i]?.[j] ?? 0;
    const gradIdx = selection.tensorId === outId && selection.index?.length === 4 ? selection.index : [n, o, Math.floor(win / W), win % W];
    return (
      <figure className="nvz-scene nvz-cnn">
        <figcaption>
          {backward ? "Max-pool backward: the gradient only flows back to the position that won" : `Max pool ${size}×${size}: each window keeps its largest value`}
          <span className="nvz-muted"> · marked cells = each window's winner{capture ? "" : " (from the forward node's argmax)"}</span>
        </figcaption>
        <div className="nvz-cnn-row">
          {backward ? (
            <>
              <ChannelGrids tensorId={node.inputs[0]} prefix={[n]} caption={`upstream gradient g, sample ${n}`} channelLabel={(c) => `map ${c}`} outline={(c, h, w) => c === o && h === i && w === j} />
              <span className="nvz-cnn-op">→</span>
              <ChannelGrids tensorId={outId} prefix={[n]} caption={`∂L/∂input, sample ${n}`} channelLabel={(c) => `map ${c}`} windows={size} winners={winners} />
            </>
          ) : (
            <>
              <ChannelGrids tensorId={inId} prefix={[n]} caption={`input, sample ${n}`} channelLabel={(c) => `map ${c}`} windows={size} winners={winners} />
              <span className="nvz-cnn-op">→</span>
              <ChannelGrids tensorId={outId} prefix={[n]} caption={`pooled, sample ${n}`} channelLabel={(c) => `map ${c}`} outline={(c, h, w) => c === o && h === i && w === j} />
            </>
          )}
        </div>
        {backward ? (
          <p className="nvz-text">
            Of the {inShape[1] * inShape[2] * inShape[3]} input positions of this sample's maps, {inShape[1] * (inShape[2] / size) * (inShape[3] / size)} won a window and receive their window's gradient; the rest are exactly 0
            ({zeros} zeros in ∂L/∂input over all {inShape[0]} sample{inShape[0] === 1 ? "" : "s"}).
            {!capture && " Click a gradient cell to see its line."}
          </p>
        ) : (
          <p className="nvz-text nvz-muted">Click a pooled cell to see its window; z{[n, o, i, j].map(sub).join(",")} = {fmt(out.values?.[((n * out.shape[1] + o) * out.shape[2] + i) * out.shape[3] + j])}.</p>
        )}
        <Worked node={node} outId={outId} index={backward ? gradIdx : idx} />
      </figure>
    );
  }
  return null;
}
