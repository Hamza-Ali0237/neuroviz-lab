# frontend/dev

Developer-only scripts for the viewer. None of this ships in the package.

## Build

```sh
cd frontend && npm install && npm run build   # writes src/nvzlab/viewer/static/nvzlab.{js,css}
```

`npm run dev` serves the viewer with `dev/sample-trace.json`.

## Screenshots

```sh
python dev/make_pages.py /tmp/nvz-pages          # exports the pages the script drives
npm run screenshots -- /tmp/nvz-pages ../../nvz-artifacts/<task>
```

Headless WebGL needs Chrome with `--use-angle=swiftshader --enable-unsafe-swiftshader`
(already in the script). The script fails on any console error.

## Capture (screen recordings)

Capture mode is the viewer with no controls, a fixed 1080×1080 (`square`) or 1080×1920
(`vertical`) stage scaled to the window, large type, auto-play through every step and a clean
loop. It is reachable from Python (`trace.save("x.html", capture="square")`,
`nvz.show(..., capture="vertical")`) and from the URL of any exported page
(`?capture=square&dim=3d&seconds=2&scene=loss`).

```sh
python -c "import nvzlab as nvz; nvz.trace('logistic_regression', kind='train', samples='batch', iterations=3).save('/tmp/lr.json')"
node dev/capture.mjs /tmp/lr.json --size vertical --dim 3d --scene loss --out ../../nvz-artifacts/captures/2026-01-01_lr-loss_see-the-math.webm
```

Writes `<out>.webm`, `<out>.png` (poster) and `<out>.mp4` when `ffmpeg` is on PATH.
Options: `--seconds` (per step, default 2), `--theme light|dark`, `--scene` (3D scene id:
`probability`, `loss`, `network`, `surface`, `regions`, `conv`, `attention`, `tensors`; 2D
tree panel: `split`, `tree`, `partition`; 2D MLP panel: `regions`), `--steps a:b` (play and
loop only steps a..b), `--poster-step k` (poster from step k; default: the first step played).
The recording starts at page load, so the first few frames are the page appearing; it ends
after the wrap back to the first step.

Playwright records video through its own `ffmpeg` download, which does not exist for every
platform (macOS 12, for one). There, `--frames` writes one PNG per step to `<out>.frames/`
(the page is paced slowly so software WebGL has time to draw), plus an `.mp4` from those
frames at the real `--seconds` pacing when `ffmpeg` is on PATH; a GIF comes from

```sh
python dev/frames_to_gif.py out.frames 2 out.gif --width 720   # needs Pillow
```
