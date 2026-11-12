import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// KaTeX ships woff2 + woff + ttf for every font. Keep woff2 only so the single CSS file
// stays small once the fonts are inlined as data URIs (the bundle must be fully offline).
const woff2Only: Plugin = {
  name: "nvz-woff2-only",
  enforce: "pre",
  transform(code, id) {
    if (!id.endsWith("katex.min.css")) return null;
    return code.replace(/,url\([^)]*\.(woff|ttf)\) format\("(woff|truetype)"\)/g, "");
  },
};

export default defineConfig({
  plugins: [react(), woff2Only],
  build: {
    outDir: "../src/nvzlab/viewer/static",
    emptyOutDir: true,
    assetsInlineLimit: 1 << 20,
    cssCodeSplit: false,
    sourcemap: false,
    rollupOptions: {
      input: "src/main.tsx",
      output: {
        inlineDynamicImports: true,
        entryFileNames: "nvzlab.js",
        assetFileNames: "nvzlab.[ext]",
      },
    },
  },
});
