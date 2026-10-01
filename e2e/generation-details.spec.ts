import { expect, test } from "@playwright/test";

const artifactId = `sha256:${"a".repeat(64)}`;
const stamp = "2026-09-30T12:00:00Z";

for (const width of [1280, 375]) {
  test.describe(`generation details at ${width} pixels`, () => {
    test.use({ viewport: { width, height: 800 } });
    for (const kind of ["image", "video"] as const) {
      test(`opens the ${kind} record by keyboard without fetching while collapsed`, async ({ page }) => {
        let reads = 0;
        const name = "Watercolor study";
        const loraName = "Watercolor texture with a long captured display name ".repeat(5).trim();
        await page.route("**/api/artifact-library?*", (route) => route.fulfill({ json: {
          items: [{
            id: `libentry:${artifactId}`, artifact_id: artifactId, version: 1, state: "visible",
            display_name: name, favorite: false, kind, media_type: `${kind}/${kind === "image" ? "png" : "mp4"}`,
            size_bytes: 1024, created_at: stamp, updated_at: stamp,
          }], next_cursor: null,
        } }));
        await page.route("**/api/artifacts/*/content", (route) => route.fulfill({ status: 204 }));
        await page.route(`**/api/artifacts/${encodeURIComponent(artifactId)}`, (route) => {
          reads += 1;
          return route.fulfill({ json: { metadata_json: { run_id: "recorded-run" } } });
        });
        await page.route("**/api/runs/recorded-run", (route) => {
          reads += 1;
          return route.fulfill({ json: { provenance_json: {
            model: { profile_name: "Recorded model" },
            workflow: { family_name: "Recorded workflow", version: 3 },
            resolved_settings: { seed: 0, steps: 20, frames: 81, fps: 24 },
            auxiliary_assets: { lora_stack: [{ name: loraName, enabled: true, model_strength: 0.7, clip_strength: 0.2 }] },
            outputs: [{ artifact_id: artifactId }],
          } } });
        });
        await page.goto("/");
        await page.getByRole("dialog", { name: "Set up LM Atelier" }).getByRole("button", { name: "Not now" }).click();
        const navigation = page.getByRole("button", { name: "Toggle navigation" });
        if (await navigation.isVisible()) await navigation.click();
        await page.locator(".primary-nav").getByRole("button", { name: "Media library", exact: true }).click();
        await expect(page.getByText(name, { exact: true })).toBeVisible();
        const summary = page.locator("summary").filter({ hasText: "Generation details" });
        await expect(summary).toBeVisible();
        expect(reads).toBe(0);
        await expect(page.getByText("Recorded model", { exact: true })).toHaveCount(0);
        await summary.focus();
        await page.keyboard.press("Enter");
        await expect(page.getByText("Recorded model", { exact: true })).toBeVisible();
        await expect(page.getByText(loraName, { exact: true })).toBeVisible();
        await expect(summary).toBeFocused();
        expect(reads).toBe(2);
        const content = page.locator(".generation-details-content");
        expect(await content.evaluate((element) => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
        await page.keyboard.press("Space");
        await expect(content).toHaveCount(0);
        await expect(summary).toBeFocused();
        await page.keyboard.press("Enter");
        await expect(page.getByText("Recorded model", { exact: true })).toBeVisible();
        expect(reads).toBe(2);
      });
    }
  });
}
