import { expect, test, type Page } from "@playwright/test";
import type { EnlargementPreview } from "../apps/web/src/studioEnlargement";

/** The browser offers only the scale a workflow takes and sends what was chosen. */
function preview(adjustable: boolean): EnlargementPreview {
  return {
    version: 1, status: "ready", workflow_revision_id: "enlarge-picture",
    fixed_factor: adjustable ? null : 4, request_authorized: false,
    factor: adjustable ? {
      key: "upscale_factor", label: "Scale", type: "enum", default: 2,
      choices: [2, 4], minimum: null, maximum: null, step: null,
      scope: "workflow", visibility: "basic", restart_required: false,
      available: true, unavailable_reason: null, help: "",
    } : null,
  };
}

async function openEnhance(page: Page) {
  await page.route("**/api/studio/capabilities", (route) => route.fulfill({
    json: { tools: [{
      kind: "enhance", workflow_class: "upscale", available: true, reason: null,
      workflow_revision_id: "enlarge-picture", adapter_asset_id: null,
    }] },
  }));
  await page.goto("/");
  const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setup).toBeVisible();
  await setup.getByRole("button", { name: "Not now" }).click();
  const navigation = page.getByRole("button", { name: "Toggle navigation" });
  if (await navigation.isVisible()) await navigation.click();
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();
  const data = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 96;
    canvas.height = 64;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("The picture canvas is unavailable.");
    context.fillStyle = "#80a0c0";
    context.fillRect(0, 0, 96, 64);
    return canvas.toDataURL("image/png").split(",")[1];
  });
  await page.getByLabel("Choose an image to edit").setInputFiles({
    name: "picture.png", mimeType: "image/png", buffer: Buffer.from(data, "base64"),
  });
  await page.getByRole("button", { name: /^Enlarge and restore detail/ }).click();
}

async function captureTurns(page: Page): Promise<Record<string, unknown>[]> {
  const turns: Record<string, unknown>[] = [];
  await page.route("**/api/chats/*/turns", async (route) => {
    turns.push(route.request().postDataJSON() as Record<string, unknown>);
    await route.fulfill({
      status: 422, json: { code: "workflow-changed", detail: "The workflow changed." },
    });
  });
  return turns;
}

for (const width of [1280, 375]) {
  test.describe(`enlargement at ${width} pixels`, () => {
    test.use({ viewport: { width, height: 800 } });

    test("chooses only factors the workflow offers and sends the chosen factor", async ({ page }) => {
      await page.route("**/api/chats/*/upscale/preview", (route) => route.fulfill({ json: preview(true) }));
      const turns = await captureTurns(page);
      await openEnhance(page);
      const choices = page.getByRole("group", { name: "Workflow scale" });
      await expect(choices).toBeVisible();
      await expect(choices.getByRole("button")).toHaveCount(2);
      await expect(choices.getByRole("button", { name: "2x", exact: true })).toHaveAttribute("aria-pressed", "true");
      const four = choices.getByRole("button", { name: "4x", exact: true });
      await four.focus();
      await page.keyboard.press("Enter");
      await expect(four).toBeFocused();
      await expect(four).toBeInViewport();
      await expect(four).toHaveAttribute("aria-pressed", "true");
      await expect(page.getByRole("slider", { name: "Workflow scale" })).toHaveCount(0);
      const apply = page.getByRole("button", { name: "Enlarge", exact: true });
      await expect(apply).toHaveAttribute("aria-disabled", "false");
      await apply.click();
      await expect.poll(() => turns.length).toBe(1);
      expect(turns[0]).toMatchObject({
        upscale: true, workflow_revision_id: "enlarge-picture", settings: { upscale_factor: 4 },
      });
      expect(turns[0].input_artifact_ids).toEqual([expect.any(String)]);
    });

    test("keeps a fixed-size workflow usable without sending an invented multiplier", async ({ page }) => {
      await page.route("**/api/chats/*/upscale/preview", (route) => route.fulfill({ json: preview(false) }));
      const turns = await captureTurns(page);
      await openEnhance(page);
      await expect(page.getByText("Enlarges 4x, set by the workflow.")).toBeVisible();
      await expect(page.getByRole("group", { name: "Workflow scale" })).toHaveCount(0);
      await expect(page.getByRole("slider", { name: "Workflow scale" })).toHaveCount(0);
      const apply = page.getByRole("button", { name: "Enlarge 4x", exact: true });
      await expect(apply).toHaveAttribute("aria-disabled", "false");
      await apply.click();
      await expect.poll(() => turns.length).toBe(1);
      expect(turns[0]).toMatchObject({ upscale: true, workflow_revision_id: "enlarge-picture" });
      expect(turns[0].settings ?? {}).toEqual({});
    });

    test("holds Apply during a refused refresh and never reuses the earlier preview", async ({ page }) => {
      let asked = 0;
      let release = () => {};
      const pending = new Promise<void>((resolve) => { release = resolve; });
      await page.route("**/api/chats/*/upscale/preview", async (route) => {
        asked += 1;
        if (asked === 1) {
          await route.fulfill({ json: preview(false) });
          return;
        }
        await pending;
        await route.fulfill({
          status: 409, json: { code: "workflow-not-ready", detail: "No enlargement workflow is ready." },
        });
      });
      const turns = await captureTurns(page);
      try {
        await openEnhance(page);
        const apply = page.getByRole("button", { name: "Enlarge 4x", exact: true });
        await expect(apply).toHaveAttribute("aria-disabled", "false");
        await apply.click();
        await expect(page.getByText("Checking what the workflow can do…")).toBeVisible();
        const waiting = page.getByRole("button", { name: "Enlarge", exact: true });
        await expect(waiting).toHaveAttribute("aria-disabled", "true");
        await expect(waiting).toBeFocused();
        await page.keyboard.press("Enter");
        expect(turns).toHaveLength(1);
        release();
        await expect(page.getByText("No enlargement workflow is ready.")).toBeVisible();
        await expect(waiting).toHaveAttribute("aria-disabled", "true");
        await page.keyboard.press("Enter");
        expect(turns).toHaveLength(1);
        expect(asked).toBe(2);
      } finally {
        release();
      }
    });
  });
}
