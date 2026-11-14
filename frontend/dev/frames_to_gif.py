"""Assemble the per-step PNG frames written by ``capture.mjs --frames`` into a looping GIF.

Usage: python frames_to_gif.py <frames-dir> <seconds-per-step> <out.gif> [--width 720]
Every frame is shown for ``seconds-per-step``; the last one one second longer, which is the
hold capture mode makes before it loops. Needs Pillow (not a project dependency; any Python
with Pillow works).
"""

import sys
from pathlib import Path

from PIL import Image

frames_dir, seconds, out = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
width = int(sys.argv[sys.argv.index("--width") + 1]) if "--width" in sys.argv else None
files = sorted(frames_dir.glob("*.png"))
if not files:
    sys.exit(f"no PNG frames in {frames_dir}")
images = []
for f in files:
    im = Image.open(f).convert("RGB")
    if width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    images.append(im.quantize(colors=256, method=Image.Quantize.MEDIANCUT))
durations = [round(seconds * 1000)] * len(images)
durations[-1] += 1000
images[0].save(
    out, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True
)
print(f"{out}: {len(images)} steps, {sum(durations) / 1000:.1f}s per loop")
