import { expect, test, type Locator, type Page } from "@playwright/test";

/** Moving and zooming the Studio's picture by hand, in a real browser.
 *
 * A drag carries the picture with the pointer the whole way, and two fingers
 * zoom it: the picture scales with their spread and keeps what lay under each
 * finger under it. The touches go through the browser's own touch input, so
 * they reach the canvas as the pointer events a touch screen sends.
 */

type Box = { x: number; y: number; width: number; height: number };
type Point = { x: number; y: number };

async function dismissSetup(page: Page) {
  const setupDialog = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setupDialog).toBeVisible();
  await setupDialog.getByRole("button", { name: "Not now" }).click();
  await expect(setupDialog).toBeHidden();
}

/** A plain picture, 200 by 100, in two colors of the caller's choosing: the
 * same picture opened twice reopens its earlier session, so each test brings its own. */
async function widePicture(page: Page, left: string, right: string): Promise<Buffer> {
  const dataUrl = await page.evaluate(([leftColor, rightColor]) => {
    const canvas = document.createElement("canvas");
    canvas.width = 200;
    canvas.height = 100;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no canvas");
    context.fillStyle = leftColor;
    context.fillRect(0, 0, 100, 100);
    context.fillStyle = rightColor;
    context.fillRect(100, 0, 100, 100);
    return canvas.toDataURL("image/png");
  }, [left, right]);
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

async function box(locator: Locator): Promise<Box> {
  const found = await locator.boundingBox();
  if (!found) throw new Error("not on screen");
  return found;
}

function picture(page: Page): Locator {
  return page.locator(".studio-canvas-layers");
}

/** Open the picture in the Studio and wait until it is fitted to the stage. */
async function openPicture(page: Page, name: string, left: string, right: string): Promise<Box> {
  await page.goto("/");
  await dismissSetup(page);
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();
  await page.getByLabel("Choose an image to edit").setInputFiles({
    name,
    mimeType: "image/png",
    buffer: await widePicture(page, left, right),
  });
  // The stage is measured again on every look: it settles as the panel beside
  // it fills in, and a size read before then is one the picture never reaches.
  await expect.poll(async () => {
    const stage = await box(page.locator(".studio-canvas"));
    const fit = Math.min(stage.width / 200, stage.height / 100);
    return Math.round((await box(picture(page))).width) === Math.round(200 * fit);
  }).toBe(true);
  return box(picture(page));
}

/** Where a point on screen falls in the picture, as shares of its width and height. */
function under(shown: Box, point: Point): Point {
  return { x: (point.x - shown.x) / shown.width, y: (point.y - shown.y) / shown.height };
}

test("carries the picture with a drag the whole way", async ({ page }) => {
  const start = await openPicture(page, "drag.png", "#8fb8de", "#4f7942");
  const x = Math.round(start.x + start.width / 2);
  const y = Math.round(start.y + start.height / 2);

  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + 60, y + 40, { steps: 6 });
  await page.mouse.up();

  // The last move is drawn on a later frame, which a loaded machine can take a moment to reach.
  await expect.poll(async () => (await box(picture(page))).x - start.x).toBeCloseTo(60, 0);
  const moved = await box(picture(page));
  expect(moved.x - start.x).toBeCloseTo(60, 0);
  expect(moved.y - start.y).toBeCloseTo(40, 0);
  expect(moved.width).toBeCloseTo(start.width, 1);
});

test.describe("on a touch screen", () => {
  test.use({ hasTouch: true });

  test("zooms with two fingers, keeping what lay under each finger under it", async ({ page }) => {
    const start = await openPicture(page, "pinch.png", "#d9a441", "#6b4c9a");
    const touch = await page.context().newCDPSession(page);
    const fingers = (type: "touchStart" | "touchMove" | "touchEnd", points: Point[]) =>
      touch.send("Input.dispatchTouchEvent", {
        type,
        touchPoints: points.map((point, id) => ({ ...point, id })),
      });
    const middle = { x: Math.round(start.x + start.width / 2), y: Math.round(start.y + start.height / 2) };
    const apart = (spread: number): Point[] => [
      { x: middle.x - spread, y: middle.y },
      { x: middle.x + spread, y: middle.y },
    ];
    const before = apart(40).map((finger) => under(start, finger));

    await fingers("touchStart", apart(40).slice(0, 1));
    await fingers("touchStart", apart(40));
    // Apart along the line between them, a step at a time, to twice the distance.
    for (let step = 1; step <= 8; step += 1) await fingers("touchMove", apart(40 + 5 * step));
    await fingers("touchEnd", []);

    await expect.poll(async () => (await box(picture(page))).width / start.width).toBeCloseTo(2, 1);
    const zoomed = await box(picture(page));
    expect(zoomed.width / start.width).toBeCloseTo(2, 1);
    apart(80).forEach((finger, index) => {
      const now = under(zoomed, finger);
      expect(now.x).toBeCloseTo(before[index].x, 2);
      expect(now.y).toBeCloseTo(before[index].y, 2);
    });
  });
});
