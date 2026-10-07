import { expect, test, type Page } from "@playwright/test";

/** Seeing what an edit changed, in a real browser.
 *
 * A flip needs no model and changes a known part of the picture: with a block
 * in one bottom corner, flipping swaps the two bottom halves and leaves the
 * top as it was. So exactly half the picture changed, and the overlay should
 * tint the bottom and leave the top clear.
 */

async function dismissSetup(page: Page) {
  const setupDialog = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setupDialog).toBeVisible();
  await setupDialog.getByRole("button", { name: "Not now" }).click();
  await expect(setupDialog).toBeHidden();
}

/** 128 by 64, one color above and a block of another in the bottom left corner. */
async function cornerPicture(page: Page): Promise<Buffer> {
  const dataUrl = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 128;
    canvas.height = 64;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no canvas");
    context.fillStyle = "#f4d35e";
    context.fillRect(0, 0, 128, 64);
    context.fillStyle = "#0d3b66";
    context.fillRect(0, 32, 64, 32);
    return canvas.toDataURL("image/png");
  });
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

/** One pixel of the overlay layer, laid over the picture while comparing. */
async function overlayPixel(page: Page, x: number, y: number): Promise<number[]> {
  return page.evaluate(
    ([px, py]) => {
      const layer = document.querySelector<HTMLCanvasElement>('.studio-canvas-layers canvas[data-layer="before"]');
      const context = layer?.getContext("2d");
      if (!layer || !context) return [];
      return Array.from(context.getImageData(px, py, 1, 1).data);
    },
    [x, y],
  );
}

test("tints the half of the picture a flip changed, and says it is half", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();
  await page.getByLabel("Choose an image to edit").setInputFiles({
    name: "corner.png",
    mimeType: "image/png",
    buffer: await cornerPicture(page),
  });
  const turn = page.getByRole("button", { name: "Rotate, straighten or flip" });
  await expect(turn).toBeEnabled();
  await turn.click();
  await page.getByRole("button", { name: "Flip horizontally" }).click();
  await expect(page.getByRole("group", { name: "Edit history" }).getByRole("button", { name: /^Result of step 1 / })).toBeVisible();

  const changed = page.getByRole("button", { name: "What changed" });
  await expect(changed).toBeVisible();
  await changed.click();
  await expect(page.getByText("50% of the picture changed.")).toBeVisible();

  // The top is as it was, so it stays clear. The bottom swapped navy for
  // yellow, a change of 231 in red: nearly the reddest tint, at opacity
  // 96 + 159 * 231/255 = 240. The canvas keeps colors premultiplied, so a
  // level either way is allowed on reading them back.
  await expect.poll(async () => (await overlayPixel(page, 100, 50))[3] ?? 0).toBeGreaterThan(0);
  for (const [x, y] of [[20, 50], [100, 50]]) {
    const [red, green, blue, alpha] = await overlayPixel(page, x, y);
    expect([red, blue]).toEqual([255, 0]);
    expect(Math.abs(green - 18)).toBeLessThanOrEqual(1);
    expect(Math.abs(alpha - 240)).toBeLessThanOrEqual(1);
  }
  expect((await overlayPixel(page, 20, 10))[3]).toBe(0);
  expect((await overlayPixel(page, 100, 10))[3]).toBe(0);

  // A split takes its place, and the overlay goes.
  await page.getByRole("button", { name: "Split" }).click();
  await expect(changed).toHaveAttribute("aria-pressed", "false");
  await expect(page.getByText("50% of the picture changed.")).toBeHidden();
});
