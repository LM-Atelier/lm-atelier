import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.LM_ATELIER_E2E_BASE_URL ?? "http://127.0.0.1:12340";
const outputDir = process.env.LM_ATELIER_E2E_OUTPUT_DIR ?? "test-results";

export default defineConfig({
  testDir: "./e2e",
  outputDir,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 45_000,
  expect: {
    timeout: 10_000,
  },
  reporter: "list",
  use: {
    baseURL,
    actionTimeout: 10_000,
    navigationTimeout: 15_000,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
