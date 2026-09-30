import { expect, test, type Page } from "@playwright/test";

/** Leaving Image Studio and coming back, in a real browser.
 *
 * A tool whose workflow is not installed sends the person to the library; the
 * selection they drew and the words they wrote should still be there when they
 * come back to the same picture, and so should they after the page reloads.
 */

async function dismissSetup(page: Page) {
  const setupDialog = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setupDialog).toBeVisible();
  await setupDialog.getByRole("button", { name: "Not now" }).click();
  await expect(setupDialog).toBeHidden();
}

/** A plain picture, sky over grass; the same picture reopens its earlier session, so each test brings its own sky. */
async function picture(page: Page, sky = "#8fb8de"): Promise<Buffer> {
  const dataUrl = await page.evaluate((skyColor) => {
    const canvas = document.createElement("canvas");
    canvas.width = 160;
    canvas.height = 100;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no canvas");
    context.fillStyle = skyColor;
    context.fillRect(0, 0, 160, 50);
    context.fillStyle = "#4f7942";
    context.fillRect(0, 50, 160, 50);
    return canvas.toDataURL("image/png");
  }, sky);
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

async function openPicture(page: Page, buffer: Buffer) {
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();
  await page.getByLabel("Choose an image to edit").setInputFiles({ name: "horizon.png", mimeType: "image/png", buffer });
}

/** Brush a stroke across the middle of the picture and write the words; returns what the selection covers. */
async function work(page: Page): Promise<string | null> {
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
  return drawn;
}

async function expectTheDraftBack(page: Page, drawn: string | null) {
  await expect(page.getByRole("button", { name: /^Brush a selection/ })).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".studio-selection-controls small")).toHaveText(drawn ?? "");
  await expect(page.getByRole("textbox", { name: /Describe the change here/ })).toHaveValue("make the sky warmer");
}

test("keeps the selection and the words while the person visits the workflow library", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  const nav = page.locator(".primary-nav");
  await openPicture(page, await picture(page));
  const drawn = await work(page);

  await nav.getByRole("button", { name: "Workflows", exact: true }).click();
  await expect(page.locator(".studio-canvas")).toHaveCount(0);
  await nav.getByRole("button", { name: "Image Studio" }).click();

  await expectTheDraftBack(page, drawn);
});

test("keeps the selection and the words through a reload of the page", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  const buffer = await picture(page, "#d9a441");
  await openPicture(page, buffer);
  const drawn = await work(page);

  await page.reload();
  // The same picture again reopens its session, and with it what was written down.
  await openPicture(page, buffer);

  await expectTheDraftBack(page, drawn);
});
