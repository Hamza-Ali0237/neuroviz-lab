import { useAppState, useDispatch } from "../state/store";
import type { StepRecord } from "../trace/types";
import { eventSummary } from "../format";

export function StepList() {
  const { trace, stepIndex } = useAppState();
  const dispatch = useDispatch();
  const groups = new Map<number, StepRecord[]>();
  for (const s of trace.steps) groups.set(s.iteration, [...(groups.get(s.iteration) ?? []), s]);
  const showIteration = trace.raw.kind === "train";
  return (
    <nav className="nvz-steps" aria-label="Steps">
      {[...groups.entries()].map(([it, steps]) => (
        <section key={it}>
          {showIteration && <h2>Iteration {it + 1}</h2>}
          <ol>
            {steps.map((s) => (
              <li key={s.index}>
                <button
                  className={`nvz-step nvz-phase-${s.phase} ${s.index === stepIndex ? "active" : ""}`}
                  onClick={() => dispatch({ type: "goto", step: s.index })}
                  aria-current={s.index === stepIndex ? "step" : undefined}
                >
                  <span className="nvz-step-no">{s.index}</span>
                  <span className="nvz-step-title">{s.title}</span>
                  <span className="nvz-step-ops">
                    {s.nodes.map((id) => trace.node(id).op).join(", ") ||
                      eventSummary(s.events.map((id) => trace.event(id).kind))}
                  </span>
                </button>
              </li>
            ))}
          </ol>
        </section>
      ))}
    </nav>
  );
}
