// Dev-only recorder for capture mode. Records one full auto-play loop of a trace and writes
// a .webm (plus .mp4 when ffmpeg is on PATH) and a poster .png.
// Usage: node dev/capture.mjs <trace.json> --size square|vertical --dim 2d|3d --out <file>
//        [--scene loss|probability|network|surface|regions|tensors] [--seconds 2] [--theme light|dark]
//        [--frames]  write one PNG per step to <out>.frames/ instead of a video (for machines where
//                    Playwright cannot record video); assemble with dev/frames_to_gif.py
// Needs the built bundle (npm run build) and Chrome. Headless WebGL: the same flags as screenshots.mjs.
import { chromium } from "playwright";
import { spawnSync } from "node:child_process";
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : dflt;
};
const tracePath = args.find((a, i) => !a.startsWith("--") && (i === 0 || !args[i - 1].startsWith("--")));
const size = opt("size", "square");
const dim = opt("dim", "2d");
const scene = opt("scene", "");
const seconds = Number(opt("seconds", "2"));
const theme = opt("theme", "light");
const out = opt("out", "");
const frames = args.includes("--frames");
// Frames mode screenshots every step once; a 3D screenshot can take seconds in software WebGL,
// so the page is paced slowly and the GIF gets the requested `seconds` per step instead.
const pageSeconds = frames ? Math.max(seconds, 8) : seconds;
if (!tracePath || !out || !["square", "vertical"].includes(size) || !["2d", "3d"].includes(dim)) {
  console.error("usage: capture.mjs <trace.json> --size square|vertical --dim 2d|3d --out <file> [--scene id] [--seconds n] [--theme light|dark]");
  process.exit(2);
}
const [w, h] = size === "square" ? [1080, 1080] : [1080, 1920];

// Build the page the same way Python does: bundle + CSS + trace JSON inlined into one file.
const staticDir = resolve(dirname(fileURLToPath(import.meta.url)), "../../src/nvzlab/viewer/static");
const [js, css, json] = await Promise.all([
  readFile(join(staticDir, "nvzlab.js"), "utf8"),
  readFile(join(staticDir, "nvzlab.css"), "utf8"),
  readFile(resolve(tracePath), "utf8"),
]);
const trace = JSON.parse(json);
const page_html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>nvzlab · ${trace.run_id}</title>
<style>${css}</style></head><body><div id="root"></div>
<script id="nvz-trace" type="application/json">${json.replace(/</g, "\\u003c")}</script>
<script type="module">${js}</script></body></html>`;
const work = await mkdtemp(join(tmpdir(), "nvz-capture-"));
const pagePath = join(work, "page.html");
await writeFile(pagePath, page_html);
const url = `${pathToFileURL(pagePath).href}?capture=${size}&dim=${dim}&seconds=${pageSeconds}${scene ? `&scene=${scene}` : ""}`;

const browser = await chromium.launch({
  channel: "chrome",
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
});
const context = await browser.newContext({
  viewport: { width: w, height: h },
  colorScheme: theme,
  ...(frames ? {} : { recordVideo: { dir: work, size: { width: w, height: h } } }),
});
const page = await context.newPage();
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(e.message));
await page.goto(url);
await page.waitForSelector(".nvz-capture");
if (dim === "3d") await page.waitForSelector(".nvz-canvas3d canvas");
await page.waitForTimeout(400);
const outBase = resolve(out).replace(/\.(webm|mp4)$/, "");
await mkdir(dirname(outBase), { recursive: true });
if (!frames) await page.screenshot({ path: `${outBase}.png` });
// One full loop: every step shown once, the hold on the last step, and the wrap to step 0.
const steps = trace.steps.length;
const looped = () => Number(document.querySelector(".nvz-capture")?.getAttribute("data-loop")) >= 1;
const state = () => {
  const el = document.querySelector(".nvz-capture");
  return [Number(el?.getAttribute("data-step")), Number(el?.getAttribute("data-loop"))];
};
const written = [`${outBase}.png`];
if (frames) {
  const dir = `${outBase}.frames`;
  await rm(dir, { recursive: true, force: true });
  await mkdir(dir);
  let seen = -1;
  for (;;) {
    const [stepNow, loops] = await page.evaluate(state);
    if (loops >= 1) break;
    if (stepNow === seen) {
      await page.waitForTimeout(100);
      continue;
    }
    await page.waitForTimeout(200); // let the scene settle after the step change
    const file = join(dir, `${String(stepNow).padStart(3, "0")}.png`);
    await page.screenshot({ path: file });
    if (seen === -1) {
      if (stepNow !== 0) errors.push(`first frame caught step ${stepNow}, not 0; raise --seconds`);
      await copyFile(file, `${outBase}.png`); // poster = step 0
    }
    const [after] = await page.evaluate(state);
    if (after !== stepNow) errors.push(`frame for step ${stepNow} finished during step ${after}; raise --seconds`);
    seen = stepNow;
  }
  await context.close();
  written.push(dir);
} else {
  await page.waitForFunction(looped, null, { timeout: (steps * seconds + 5) * 1000 + 10_000, polling: 100 });
  await page.waitForTimeout(300);
  const video = page.video();
  await context.close();
  const webm = `${outBase}.webm`;
  await copyFile(await video.path(), webm);
  written.push(webm);
}
await browser.close();
await rm(work, { recursive: true, force: true });
const webm = written.find((f) => f.endsWith(".webm"));
const ffmpeg = spawnSync("ffmpeg", ["-version"], { stdio: "ignore" });
if (webm && ffmpeg.status === 0) {
  const mp4 = `${outBase}.mp4`;
  const r = spawnSync("ffmpeg", ["-y", "-i", webm, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4], { stdio: "ignore" });
  if (r.status === 0) written.push(mp4);
  else console.warn("ffmpeg failed; keeping the .webm only");
} else if (webm) console.log("ffmpeg not found; wrote .webm only");
console.log(`${basename(tracePath)} → ${size} ${dim}${scene ? ` ${scene}` : ""}: ${steps} steps × ${seconds}s`);
for (const f of written) console.log(`  ${f}`);
if (errors.length) {
  console.error("console errors:\n" + errors.join("\n"));
  process.exit(1);
}
