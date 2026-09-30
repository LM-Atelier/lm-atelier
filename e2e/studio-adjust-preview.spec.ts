import { expect, test, type Page } from "@playwright/test";

/** The light and color preview is the picture Apply makes, pixel for pixel.
 *
 * The browser draws the preview while the sliders move and the server makes the
 * kept picture, each with its own copy of the arithmetic. Unit tests hold both
 * copies to the same sample pixels; this holds the whole path to it, in a real
 * browser: decode, preview on the canvas, apply, store, and read back.
 */

async function dismissSetup(page: Page) {
  const setupDialog = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setupDialog).toBeVisible();
  await setupDialog.getByRole("button", { name: "Not now" }).click();
  await expect(setupDialog).toBeHidden();
}

/** A plain opaque picture: a gradient, a square and a circle. */
async function neutralPicture(page: Page): Promise<Buffer> {
  const dataUrl = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 96;
    canvas.height = 64;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no canvas");
    const gradient = context.createLinearGradient(0, 0, 96, 64);
    gradient.addColorStop(0, "#3a6ea5");
    gradient.addColorStop(1, "#f2c14e");
    context.fillStyle = gradient;
    context.fillRect(0, 0, 96, 64);
    context.fillStyle = "#e08a2c";
    context.fillRect(8, 8, 24, 24);
    context.fillStyle = "#3aa35a";
    context.beginPath();
    context.arc(66, 36, 18, 0, Math.PI * 2);
    context.fill();
    return canvas.toDataURL("image/png");
  });
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

/** Every tone from black to white, above a dark-to-light band of color.
 *
 * A picture of its own, so it opens a studio session of its own too: the same
 * picture again would reopen the history another test left.
 */
async function tonesPicture(page: Page): Promise<Buffer> {
  const dataUrl = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 96;
    canvas.height = 64;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no canvas");
    const greys = context.createLinearGradient(0, 0, 96, 0);
    greys.addColorStop(0, "#000000");
    greys.addColorStop(1, "#ffffff");
    context.fillStyle = greys;
    context.fillRect(0, 0, 96, 32);
    const colors = context.createLinearGradient(0, 0, 96, 0);
    colors.addColorStop(0, "#1d3557");
    colors.addColorStop(1, "#f1c4a8");
    context.fillStyle = colors;
    context.fillRect(0, 32, 96, 32);
    return canvas.toDataURL("image/png");
  });
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

/** A cutout: two shapes on a transparent background.
 *
 * Whole-pixel rectangles only, so every pixel is wholly opaque or wholly
 * clear. A canvas keeps its colors premultiplied, and a partly transparent
 * pixel would not read back exactly as it was written.
 */
async function cutoutPicture(page: Page): Promise<Buffer> {
  const dataUrl = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 48;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no canvas");
    context.fillStyle = "#e08a2c";
    context.fillRect(8, 8, 30, 24);
    context.fillStyle = "#3aa35a";
    context.fillRect(28, 18, 26, 22);
    return canvas.toDataURL("image/png");
  });
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

/** Muted colors across the whole picture, with one vivid square in it. */
async function mutedPicture(page: Page): Promise<Buffer> {
  const dataUrl = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 96;
    canvas.height = 64;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no canvas");
    const muted = context.createLinearGradient(0, 0, 96, 64);
    muted.addColorStop(0, "#8a7f74");
    muted.addColorStop(1, "#6f8a9c");
    context.fillStyle = muted;
    context.fillRect(0, 0, 96, 64);
    context.fillStyle = "#d8342a";
    context.fillRect(36, 20, 24, 24);
    return canvas.toDataURL("image/png");
  });
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

/** Open `picture` in the studio with the light and color tool, once it shows. */
async function openToAdjust(page: Page, name: string, picture: Buffer, size: number): Promise<number[]> {
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();
  await page.getByLabel("Choose an image to edit").setInputFiles({ name, mimeType: "image/png", buffer: picture });
  await page.getByRole("button", { name: "Adjust light and color" }).click();
  await expect.poll(() => shownPixels(page).then((pixels) => pixels.length)).toBe(size * 4);
  return shownPixels(page);
}

/** Apply the sliders as they stand and read back the pixels of the picture kept. */
async function applyAndReadKept(page: Page): Promise<number[]> {
  await page.getByRole("button", { name: "Apply adjustments" }).click();
  await expect.poll(() => page.evaluate(async () => {
    const { id } = JSON.parse(localStorage.getItem("local-lm-studio-session") ?? "{}") as { id?: string };
    const session = (await (await fetch(`/api/studio/sessions/${id}`)).json()) as { messages?: unknown[] };
    return session.messages?.length ?? 0;
  })).toBe(2);
  return page.evaluate(async () => {
    const stored = localStorage.getItem("local-lm-studio-session");
    const { id } = JSON.parse(stored ?? "{}") as { id?: string };
    const session = (await (await fetch(`/api/studio/sessions/${id}`)).json()) as {
      messages: { role: string; parts: { type: string; artifact_id: string | null }[] }[];
    };
    const answer = session.messages.filter((message) => message.role === "assistant").at(-1);
    const image = answer?.parts.find((part) => part.type === "image");
    const blob = await (await fetch(`/api/artifacts/${image?.artifact_id}/content`)).blob();
    const bitmap = await createImageBitmap(blob);
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d");
    if (!context) return [];
    context.drawImage(bitmap, 0, 0);
    return Array.from(context.getImageData(0, 0, bitmap.width, bitmap.height).data);
  });
}

/** The pixels the canvas is showing for the picture itself. */
async function shownPixels(page: Page): Promise<number[]> {
  return page.evaluate(() => {
    // The picture's own layer is the one canvas that is not an overlay.
    const layer = document.querySelector<HTMLCanvasElement>(".studio-canvas-layers canvas:not([data-layer])");
    const context = layer?.getContext("2d");
    if (!layer || !context) return [];
    return Array.from(context.getImageData(0, 0, layer.width, layer.height).data);
  });
}

test("shows exactly the picture that applying the adjustment keeps", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  const before = await openToAdjust(page, "neutral-shapes.png", await neutralPicture(page), 96 * 64);
  await page.getByRole("slider", { name: "Saturation" }).fill("-41");
  await page.getByRole("slider", { name: "Warmth" }).fill("30");
  await page.getByRole("slider", { name: "Brightness" }).fill("15");
  await page.getByRole("slider", { name: "Tint" }).fill("-35");
  await page.getByRole("slider", { name: "Sharpness" }).fill("45");
  // The preview is drawn on the next frame after the sliders settle.
  await expect.poll(async () => (await shownPixels(page)).join() !== before.join()).toBe(true);
  const preview = await shownPixels(page);

  const kept = await applyAndReadKept(page);

  expect(kept.length).toBe(preview.length);
  const differing = kept.filter((value, index) => value !== preview[index]).length;
  expect(differing).toBe(0);
});

test("lifts the shadows and holds back the highlights exactly as the preview shows", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  const before = await openToAdjust(page, "neutral-tones.png", await tonesPicture(page), 96 * 64);
  await page.getByRole("slider", { name: "Shadows" }).fill("70");
  await page.getByRole("slider", { name: "Highlights" }).fill("-55");
  await expect.poll(async () => (await shownPixels(page)).join() !== before.join()).toBe(true);
  const preview = await shownPixels(page);

  const kept = await applyAndReadKept(page);

  expect(kept.length).toBe(preview.length);
  const differing = kept.filter((value, index) => value !== preview[index]).length;
  expect(differing).toBe(0);
});

test("sharpens a cutout exactly as the preview shows, keeping it cut out", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  const before = await openToAdjust(page, "neutral-cutout.png", await cutoutPicture(page), 64 * 48);
  await page.getByRole("slider", { name: "Sharpness" }).fill("80");
  await expect.poll(async () => (await shownPixels(page)).join() !== before.join()).toBe(true);
  const preview = await shownPixels(page);

  const kept = await applyAndReadKept(page);

  expect(kept.length).toBe(preview.length);
  const differing = kept.filter((value, index) => value !== preview[index]).length;
  expect(differing).toBe(0);
  const clear = (pixels: number[]) => pixels.filter((value, index) => index % 4 === 3 && value === 0).length;
  expect(clear(kept)).toBe(clear(before));
  expect(clear(before)).toBeGreaterThan(0);
});

test("richens muted colors and darkens the edges exactly as the preview shows", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  const before = await openToAdjust(page, "neutral-muted.png", await mutedPicture(page), 96 * 64);
  await page.getByRole("slider", { name: "Vibrance" }).fill("60");
  await page.getByRole("slider", { name: "Vignette" }).fill("70");
  await expect.poll(async () => (await shownPixels(page)).join() !== before.join()).toBe(true);
  const preview = await shownPixels(page);

  const kept = await applyAndReadKept(page);

  expect(kept.length).toBe(preview.length);
  const differing = kept.filter((value, index) => value !== preview[index]).length;
  expect(differing).toBe(0);
  // The corners went darker and the middle of the picture kept its light.
  const light = (pixels: number[], x: number, y: number) => {
    const at = (y * 96 + x) * 4;
    return pixels[at] + pixels[at + 1] + pixels[at + 2];
  };
  expect(light(kept, 0, 0)).toBeLessThan(light(before, 0, 0));
  expect(light(kept, 95, 63)).toBeLessThan(light(before, 95, 63));
  expect(light(kept, 20, 32)).toBeGreaterThan(light(kept, 0, 32));
});
