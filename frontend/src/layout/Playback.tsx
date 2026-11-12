import { useAppState, useDispatch } from "../state/store";

export function Playback() {
  const { trace, stepIndex, playing, speed } = useAppState();
  const dispatch = useDispatch();
  const last = trace.steps.length - 1;
  const step = trace.steps[stepIndex];
  return (
    <footer className="nvz-playback">
      <button onClick={() => dispatch({ type: "goto", step: 0 })} title="First step" aria-label="First step">
        ⏮
      </button>
      <button onClick={() => dispatch({ type: "prev" })} title="Previous step (←)" aria-label="Previous step">
        ◀
      </button>
      <button
        className="nvz-play"
        onClick={() => dispatch({ type: "toggle" })}
        title={playing ? "Pause (space)" : "Play (space)"}
        aria-label={playing ? "Pause" : "Play"}
      >
        {playing ? "❚❚" : "▶"}
      </button>
      <button onClick={() => dispatch({ type: "next" })} title="Next step (→)" aria-label="Next step">
        ▶|
      </button>
      <button onClick={() => dispatch({ type: "goto", step: last })} title="Last step" aria-label="Last step">
        ⏭
      </button>
      <input
        className="nvz-scrub"
        type="range"
        min={0}
        max={last}
        value={stepIndex}
        onChange={(e) => dispatch({ type: "goto", step: Number(e.target.value) })}
        aria-label="Step scrubber"
      />
      <span className={`nvz-step-label nvz-phase-${step.phase}`}>
        <b>
          {stepIndex} / {last}
        </b>{" "}
        {step.title}
      </span>
      <label className="nvz-speed">
        speed
        <select value={speed} onChange={(e) => dispatch({ type: "speed", speed: Number(e.target.value) })}>
          <option value={0.5}>0.5×</option>
          <option value={1}>1×</option>
          <option value={1.5}>1.5×</option>
          <option value={2}>2×</option>
        </select>
      </label>
    </footer>
  );
}
