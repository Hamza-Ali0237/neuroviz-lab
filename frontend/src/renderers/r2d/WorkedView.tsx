import { useAppState } from "../../state/store";
import type { NodeRecord } from "../../trace/types";
import { fmt, fmtFull } from "../../format";
import { worked } from "./worked";

/** Worked computation for one output cell of a node: the selected cell, a given one, or the first. */
export function Worked({ node, outId: givenOut, index: givenIndex }: { node: NodeRecord; outId?: string; index?: number[] }) {
  const { trace, selection, sampleIndex } = useAppState();
  const selectedHere = !!(selection.tensorId && selection.index && node.outputs.includes(selection.tensorId));
  const outId = givenOut ?? (selectedHere ? selection.tensorId! : node.outputs[0]);
  if (!outId) return null;
  const t = trace.tensor(outId);
  const defaultIndex = t.shape.map((d, i) => (i === 0 && d === trace.raw.mode.n_samples && t.role !== "param" ? sampleIndex : 0));
  const index = givenIndex ?? (selectedHere ? selection.index! : defaultIndex);
  const lines = worked(trace, node, outId, index);
  const where = `${trace.label(outId)}[${index.join(", ")}]`;
  return (
    <div className="nvz-worked">
      <h4>
        Worked computation <span className="nvz-muted">for {where}</span>
        {!selectedHere && !givenIndex && <span className="nvz-muted"> · click any output cell to change</span>}
      </h4>
      {lines ? (
        lines.map((l, i) => (
          <p key={i} className={`mono nvz-worked-line ${l.derived ? "derived" : ""}`}>
            <span className="lhs">{l.lhs}</span> = {l.expr} = <b title={l.derived ? "computed here from trace values" : `trace value ${fmtFull(l.value)}`}>{fmt(l.value)}</b>
            {l.derived && <span className="nvz-muted"> (computed here)</span>}
          </p>
        ))
      ) : (
        <p className="nvz-muted">No worked arithmetic rule for “{node.op}”; the values above are from the trace.</p>
      )}
    </div>
  );
}
