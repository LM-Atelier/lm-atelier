import { expect, test, type Page } from "@playwright/test";

/** Leaving Image Studio for the workflow library and coming back, in a real browser.
 *
 * A tool whose workflow is not installed sends the person to the library; the
 * selection they drew and the words they wrote should still be there when they
 * come back to the same picture.
 */

async function dismissSetup(page: Page) {
  const setupDialog = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setupDialog).toBeVisible();
  await setupDialog.getByRole("button", { name: "Not now" }).click();
  await expect(setupDialog).toBeHidden();
}

async function picture(page: Page): Promise<Buffer> {
  const dataUrl = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 160;
    canvas.height = 100;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no canvas");
    context.fillStyle = "#8fb8de";
    context.fillRect(0, 0, 160, 50);
    context.fillStyle = "#4f7942";
    context.fillRect(0, 50, 160, 50);
    return canvas.toDataURL("image/png");
  });
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

test("keeps the selection and the words while the person visits the workflow library", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  const nav = page.locator(".primary-nav");
  await nav.getByRole("button", { name: "Image Studio" }).click();
  await page.getByLabel("Choose an image to edit").setInputFiles({
    name: "horizon.png",
    mimeType: "image/png",
    buffer: await picture(page),
  });
  const select = page.getByRole("button", { name: /^Select part of the picture/ });
  await expect(select).toBeEnabled();
  await select.click();
  await page.getByRole("button", { name: /^Brush a selection/ }).click();

  // A stroke across the middle of the picture.
  const layers = await page.locator(".studio-canvas-layers").boundingBox();
  if (!layers) throw new Error("no picture on screen");
  await page.mouse.move(layers.x + layers.width * 0.3, layers.y + layers.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(layers.x + layers.width * 0.7, layers.y + layers.height * 0.5, { steps: 10 });
  await page.mouse.up();
  const coverage = page.locator(".studio-selection-controls small");
  await expect(coverage).toHaveText(/% of the image selected/);
  const drawn = await coverage.textContent();
  await page.getByRole("textbox", { name: /Describe the change here/ }).fill("make the sky warmer");

  await nav.getByRole("button", { name: "Workflows", exact: true }).click();
  await expect(page.locator(".studio-canvas")).toHaveCount(0);
  await nav.getByRole("button", { name: "Image Studio" }).click();

  await expect(page.getByRole("button", { name: /^Brush a selection/ })).toHaveAttribute("aria-pressed", "true");
  await expect(coverage).toHaveText(drawn ?? "");
  await expect(page.getByRole("textbox", { name: /Describe the change here/ })).toHaveValue("make the sky warmer");
});
