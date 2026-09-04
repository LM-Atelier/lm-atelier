import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, it } from "vitest";

it("keeps turn confirmation out of browser globals", () => {
  const apiSource = readFileSync(join(__dirname, "api.ts"), "utf8");
  const appSource = readFileSync(join(__dirname, "App.tsx"), "utf8");
  expect(apiSource).not.toContain("window.confirm");
  expect(apiSource).toContain("confirmTurn?: TurnConfirmationHandler");
  expect(appSource).toContain("useTurnConfirmation()");
});
