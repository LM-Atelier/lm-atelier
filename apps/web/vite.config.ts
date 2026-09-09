import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:12340",
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          icons: ["lucide-react"],
          markdown: ["react-markdown", "remark-gfm"],
          query: ["@tanstack/react-query"],
          react: ["react", "react-dom"],
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    // Vitest 4 stopped clearing recorded calls as part of restoreAllMocks, which
    // only restores original implementations. Suites here mock a module and then
    // assert that a function was never called, so without this a call made by an
    // earlier test is still on the record when the next one reads it.
    clearMocks: true,
  },
});
