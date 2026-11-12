// Dev-only visual check: open exported viewer pages, step through them and screenshot.
// Usage: npm run screenshots -- <pages-dir> <out-dir>
// Fails if any page logs a console error.
import { chromium } from "playwright";
import { readdirSync } from "node:fs";
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
const browser = await chromium.launch({ channel: "chrome" });
const errors = [];

for (const file of readdirSync(pagesDir).filter((f) => f.endsWith(".html")).sort()) {
  const name = file.replace(/\.html$/, "");
  for (const [w, h] of SIZES) {
    for (const theme of THEMES) {
      const page = await browser.newPage({ viewport: { width: w, height: h }, colorScheme: theme });
      page.on("console", (m) => m.type() === "error" && errors.push(`${file}: ${m.text()}`));
      page.on("pageerror", (e) => errors.push(`${file}: ${e.message}`));
      await page.goto(pathToFileURL(join(pagesDir, file)).href);
      await page.waitForSelector(".nvz-app");
      const last = await page.locator(".nvz-steps .nvz-step").count() - 1;
      const shot = async (tag) => {
        await page.locator(".nvz-centre").evaluate((el) => el.scrollTo(0, 0));
        await page.screenshot({ path: join(outDir, `${name}__${tag}__${w}x${h}__${theme}.png`) });
      };
      await shot("first");
      // Middle step: the last step with a sigmoid or the opaque op, then select its last output cell.
      const ops = await page.locator(".nvz-steps .nvz-step-ops").allTextContents();
      let mid = Math.max(1, Math.floor(last / 2));
      ops.forEach((t, i) => /sigmoid|clamp_half/.test(t) && (mid = i));
      await page.locator(".nvz-steps .nvz-step").nth(mid).click();
      const cells = page.locator(".nvz-tensor-group").last().locator(".nvz-cell");
      if ((await cells.count()) > 0) await cells.last().click();
      await page.waitForTimeout(300);
      await shot("middle-selected");
      await page.locator(".nvz-steps .nvz-step").nth(last).click();
      await page.waitForTimeout(300);
      await shot("last");
      await page.close();
    }
  }
  console.log(`done ${file}`);
}
await browser.close();
if (errors.length) {
  console.error("console errors:\n" + errors.join("\n"));
  process.exit(1);
}
console.log("no console errors");
