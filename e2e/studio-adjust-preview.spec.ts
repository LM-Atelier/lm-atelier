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
  const picture = await neutralPicture(page);
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();
  await page.getByLabel("Choose an image to edit").setInputFiles({
    name: "neutral-shapes.png",
    mimeType: "image/png",
    buffer: picture,
  });

  await page.getByRole("button", { name: "Adjust light and color" }).click();
  await expect.poll(() => shownPixels(page).then((pixels) => pixels.length)).toBe(96 * 64 * 4);
  const before = await shownPixels(page);
  await page.getByRole("slider", { name: "Saturation" }).fill("-41");
  await page.getByRole("slider", { name: "Warmth" }).fill("30");
  await page.getByRole("slider", { name: "Brightness" }).fill("15");
  await page.getByRole("slider", { name: "Tint" }).fill("-35");
  // The preview is drawn on the next frame after the sliders settle.
  await expect.poll(async () => (await shownPixels(page)).join() !== before.join()).toBe(true);
  const preview = await shownPixels(page);

  await page.getByRole("button", { name: "Apply adjustments" }).click();
  await expect.poll(() => page.evaluate(async () => {
    const { id } = JSON.parse(localStorage.getItem("local-lm-studio-session") ?? "{}") as { id?: string };
    const session = (await (await fetch(`/api/studio/sessions/${id}`)).json()) as { messages?: unknown[] };
    return session.messages?.length ?? 0;
  })).toBe(2);
  const kept = await page.evaluate(async () => {
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

  expect(kept.length).toBe(preview.length);
  const differing = kept.filter((value, index) => value !== preview[index]).length;
  expect(differing).toBe(0);
});
