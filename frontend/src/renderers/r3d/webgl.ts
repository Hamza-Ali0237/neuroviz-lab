/** WebGL availability, probed once. The 3D toggle is disabled with the reason when it fails. */
let cached: { ok: boolean; reason: string } | null = null;

export function webglSupport(): { ok: boolean; reason: string } {
  if (cached) return cached;
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
    cached = gl ? { ok: true, reason: "" } : { ok: false, reason: "this browser did not provide a WebGL context" };
  } catch (err) {
    cached = { ok: false, reason: `WebGL failed to start (${String(err)})` };
  }
  return cached;
}
