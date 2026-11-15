// Dev-only visual check: open exported viewer pages, drive them and screenshot.
// Usage: npm run screenshots -- <pages-dir> <out-dir>
// Expects pages written by frontend/dev/make_pages.py: lr-predict-f2.html, lr-train-batch-f2.html,
// opaque.html, mlp-predict.html, mlp-train-batch.html, tree-predict.html, tree-train.html, cnn-train.html, transformer-train.html. Fails if any page logs a console error.
// Headless WebGL: Chrome with --use-angle=swiftshader --enable-unsafe-swiftshader.
import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [pagesDir, outDir] = process.argv.slice(2).map((p) => resolve(p));
if (!pagesDir || !outDir) throw new Error("usage: screenshots.mjs <pages-dir> <out-dir>");
await mkdir(outDir, { recursive: true });

const SIZES = [
  [1440, 900],
  [1024, 768],
];
const THEMES = ["light", "dark"];
const browser = await chromium.launch({
  channel: "chrome",
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
});
const errors = [];

async function open(file, w, h, theme) {
  const page = await browser.newPage({ viewport: { width: w, height: h }, colorScheme: theme });
  page.on("console", (m) => m.type() === "error" && errors.push(`${file}: ${m.text()}`));
  page.on("pageerror", (e) => errors.push(`${file}: ${e.message}`));
  await page.goto(pathToFileURL(join(pagesDir, file)).href);
  await page.waitForSelector(".nvz-app");
  return page;
}
const shot = (page, name, w, h, theme) =>
  page.screenshot({ path: join(outDir, `${name}__${w}x${h}__${theme}.png`) });
const to3d = async (page) => {
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.waitForSelector(".nvz-canvas3d canvas");
  await page.waitForTimeout(600);
};
const gotoStep = (page, i) => page.locator(".nvz-steps .nvz-step").nth(i).click();

for (const [w, h] of SIZES) {
  for (const theme of THEMES) {
    // (a) LR predict f2, first step, 2D: context dataset and legends. (b) same in 3D.
    let page = await open("lr-predict-f2.html", w, h, theme);
    await shot(page, "a-lr-predict-f2-step0-2d", w, h, theme);
    await to3d(page);
    await shot(page, "b-lr-predict-f2-step0-3d-surface", w, h, theme);
    await page.close();

    // (c) LR train batch f2, update step of iteration 2, 3D loss landscape with path.
    page = await open("lr-train-batch-f2.html", w, h, theme);
    const titles = await page.locator(".nvz-steps .nvz-step-title").allTextContents();
    const updates = titles.map((t, i) => (t.startsWith("Update") ? i : -1)).filter((i) => i >= 0);
    await gotoStep(page, updates[1]);
    await to3d(page);
    await shot(page, "c-lr-train-f2-update2-3d-loss", w, h, theme);
    // Select a cell in 3D through the tensor stack scene (at an update step it shows the
    // updated params too), then switch to 2D: the selection must survive the renderer switch.
    await page.locator(".nvz-3d-tools select").selectOption("tensors");
    await page.waitForTimeout(400);
    await page.locator(".nvz-3d-value").last().click();
    await page.waitForTimeout(300);
    const selected = await page.locator(".nvz-inspector dd").count();
    if (!selected) errors.push("c: clicking the 3D cell did not select anything");
    // (d) back to 2D with the selection intact.
    await page.getByRole("button", { name: "2D", exact: true }).click();
    await page.waitForTimeout(300);
    await shot(page, "d-lr-train-f2-update2-2d-selected", w, h, theme);
    await page.close();

    // (e) opaque example in 3D: generic tensor stack.
    page = await open("opaque.html", w, h, theme);
    await gotoStep(page, 1);
    await to3d(page);
    await shot(page, "e-opaque-3d-tensors", w, h, theme);
    await page.close();

    // (f) MLP predict, softmax step, 2D network diagram + decision regions; (g) 3D network.
    page = await open("mlp-predict.html", w, h, theme);
    await gotoStep(page, 4);
    await shot(page, "f-mlp-predict-softmax-2d-network", w, h, theme);
    await to3d(page);
    await shot(page, "g-mlp-predict-3d-network", w, h, theme);
    await page.locator(".nvz-3d-tools select").selectOption("surface");
    await page.waitForTimeout(600);
    await shot(page, "h-mlp-predict-3d-surface", w, h, theme);
    await page.close();

    // (i) MLP train, gradients reach W1: edges show gradients.
    page = await open("mlp-train-batch.html", w, h, theme);
    await gotoStep(page, 11);
    await shot(page, "i-mlp-train-backward-2d-gradients", w, h, theme);
    await page.close();

    // (j) tree predict, third node test: path highlighted; (k) 3D leaf regions.
    page = await open("tree-predict.html", w, h, theme);
    await gotoStep(page, 3);
    await shot(page, "j-tree-predict-path-2d", w, h, theme);
    await to3d(page);
    await shot(page, "k-tree-predict-3d-regions", w, h, theme);
    await page.close();

    // (l) tree train, root split search; (m) a deeper split with the partition grown.
    page = await open("tree-train.html", w, h, theme);
    await shot(page, "l-tree-train-root-split-2d", w, h, theme);
    await gotoStep(page, 5);
    await shot(page, "m-tree-train-split5-2d", w, h, theme);
    await to3d(page);
    await shot(page, "n-tree-train-3d-regions", w, h, theme);
    await page.close();

    // (o) CNN train: convolution worked sum for z[0,0,0,0]; (p) max pool winners; (q) pool backward routing;
    // (r) 3D convolution stack with the receptive-field box.
    page = await open("cnn-train.html", w, h, theme);
    await gotoStep(page, 1);
    await shot(page, "o-cnn-train-conv-worked-2d", w, h, theme);
    await gotoStep(page, 3);
    await shot(page, "p-cnn-train-maxpool-2d", w, h, theme);
    await gotoStep(page, 13);
    await shot(page, "q-cnn-train-pool-backward-2d", w, h, theme);
    await gotoStep(page, 1);
    await to3d(page);
    await page.locator(".nvz-3d-tools select").selectOption("conv");
    await page.waitForTimeout(600);
    await shot(page, "r-cnn-train-3d-conv", w, h, theme);
    await page.close();

    // (s) Transformer train: embedding rows; (t) attention map with the second row's worked softmax;
    // (u) 3D attention bars.
    page = await open("transformer-train.html", w, h, theme);
    await gotoStep(page, 1);
    await shot(page, "s-transformer-train-embedding-2d", w, h, theme);
    await gotoStep(page, 9);
    await page.locator(".nvz-attn tbody tr").nth(1).click();
    await page.waitForTimeout(200);
    if (!(await page.locator(".nvz-softmax-row").count())) errors.push("t: clicking an attention row did not show its worked softmax");
    await shot(page, "t-transformer-train-attention-2d", w, h, theme);
    await to3d(page);
    await page.locator(".nvz-3d-tools select").selectOption("attention");
    await page.waitForTimeout(600);
    await shot(page, "u-transformer-train-3d-attention", w, h, theme);
    await page.close();
  }
  console.log(`done ${w}x${h}`);
}
await browser.close();
if (errors.length) {
  console.error("console errors:\n" + errors.join("\n"));
  process.exit(1);
}
console.log("no console errors");
