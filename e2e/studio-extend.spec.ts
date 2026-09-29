import { expect, test, type Locator, type Page } from "@playwright/test";

/** Extending a picture in a real browser: the frame follows the picture as it is shown.
 *
 * The canvas fits the picture to the stage, so on screen it is seldom its own
 * size, and it can be zoomed. A drag on an edge has to ask for the same share
 * of the picture whatever the zoom, and the frame has to be drawn where the
 * extended canvas will be, so these measure both against the picture's box on
 * the page rather than against its pixels.
 */

type Box = { x: number; y: number; width: number; height: number };

async function dismissSetup(page: Page) {
  const setupDialog = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setupDialog).toBeVisible();
  await setupDialog.getByRole("button", { name: "Not now" }).click();
  await expect(setupDialog).toBeHidden();
}

/** A plain tall picture, 100 by 160, so the fitted view leaves room at either side. */
async function tallPicture(page: Page): Promise<Buffer> {
  const dataUrl = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 100;
    canvas.height = 160;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no canvas");
    context.fillStyle = "#8fb8de";
    context.fillRect(0, 0, 100, 80);
    context.fillStyle = "#4f7942";
    context.fillRect(0, 80, 100, 80);
    return canvas.toDataURL("image/png");
  });
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

async function openExtend(page: Page) {
  await page.goto("/");
  await dismissSetup(page);
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();
  await page.getByLabel("Choose an image to edit").setInputFiles({
    name: "tall.png",
    mimeType: "image/png",
    buffer: await tallPicture(page),
  });
  const extend = page.getByRole("button", { name: /^Extend past the edge/ });
  await expect(extend).toBeEnabled();
  await extend.click();
  await expect(page.locator('.studio-extend-handle[data-side="right"]')).toBeVisible();
}

async function box(locator: Locator): Promise<Box> {
  const found = await locator.boundingBox();
  if (!found) throw new Error("not on screen");
  return found;
}

/** Press the middle of an edge's grip and move by (dx, dy) screen pixels. */
async function drag(page: Page, side: string, dx: number, dy: number) {
  const grip = await box(page.locator(`.studio-extend-handle[data-side="${side}"]`));
  const x = grip.x + grip.width / 2;
  const y = grip.y + grip.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + dx, y + dy, { steps: 8 });
  await page.mouse.up();
}

function picture(page: Page): Locator {
  return page.locator(".studio-canvas-layers");
}

async function expectFrameAround(page: Page, shown: Box, gained: { top: number; right: number; bottom: number; left: number }) {
  const canvas = await box(page.locator(".studio-canvas"));
  const across = shown.width / 100;
  const down = shown.height / 160;
  const edge = page.locator(".studio-extend-edge");
  const read = async (name: string) => Number(await edge.getAttribute(name));
  expect(await read("x")).toBeCloseTo(shown.x - canvas.x - gained.left * across, 0);
  expect(await read("y")).toBeCloseTo(shown.y - canvas.y - gained.top * down, 0);
  expect(await read("width")).toBeCloseTo(shown.width + (gained.left + gained.right) * across, 0);
  expect(await read("height")).toBeCloseTo(shown.height + (gained.top + gained.bottom) * down, 0);
}

test("asks for the same share of the picture at the fitted zoom and zoomed in", async ({ page }) => {
  await openExtend(page);
  const fitted = await box(picture(page));
  // Fitted to the stage the picture is not its own size, which is what a
  // drag measured in its pixels got wrong.
  expect(Math.abs(fitted.width - 100)).toBeGreaterThan(20);

  await drag(page, "right", fitted.width / 4, 0);
  await expect(page.getByRole("button", { name: "Extend to the right, currently 25 percent" })).toBeVisible();
  await expect(page.getByText("The canvas goes from 100 × 160 to 125 × 160.")).toBeVisible();
  await expectFrameAround(page, fitted, { top: 0, right: 25, bottom: 0, left: 0 });

  // Zoom in about the middle of the view and drag the other edge.
  await page.locator(".studio-canvas").focus();
  await page.keyboard.press("+");
  const zoomed = await box(picture(page));
  expect(zoomed.width).toBeCloseTo(fitted.width * 1.2, 0);

  await drag(page, "left", -zoomed.width / 4, 0);
  await expect(page.getByRole("button", { name: "Extend to the left, currently 25 percent" })).toBeVisible();
  await expect(page.getByText("The canvas goes from 100 × 160 to 150 × 160.")).toBeVisible();
  await expectFrameAround(page, zoomed, { top: 0, right: 25, bottom: 0, left: 25 });
});

test("keeps an edge's grip in view after the edge goes past the view", async ({ page }) => {
  await openExtend(page);
  const shown = await box(picture(page));

  const top = page.locator('.studio-extend-handle[data-side="top"]');
  await top.focus();
  for (let press = 0; press < 20; press += 1) await page.keyboard.press("ArrowDown");
  await expect(page.getByRole("button", { name: "Extend upward, currently 100 percent" })).toBeVisible();

  // The new top edge is a whole picture's height above the picture, out of
  // the view; its grip stays inside the canvas, where it can still be taken.
  const canvas = await box(page.locator(".studio-canvas"));
  const grip = await box(top);
  expect(grip.y).toBeGreaterThanOrEqual(canvas.y - 0.5);
  expect(grip.y + grip.height).toBeLessThanOrEqual(canvas.y + canvas.height + 0.5);
  await drag(page, "top", 0, shown.height / 2);
  await expect(page.getByRole("button", { name: "Extend upward, currently 50 percent" })).toBeVisible();
});

test.describe("on a display with two device pixels to a CSS pixel", () => {
  test.use({ deviceScaleFactor: 2 });

  test("asks for the same share of the picture", async ({ page }) => {
    await openExtend(page);
    const shown = await box(picture(page));

    await drag(page, "bottom", 0, shown.height / 4);
    await expect(page.getByRole("button", { name: "Extend downward, currently 25 percent" })).toBeVisible();
    await expect(page.getByText("The canvas goes from 100 × 160 to 100 × 200.")).toBeVisible();
    await expectFrameAround(page, shown, { top: 0, right: 0, bottom: 40, left: 0 });
  });
});
