import { expect, test, type Page } from "@playwright/test";

/** Added words look on the canvas as they come out, in a real browser with real fonts.
 *
 * The browser draws the words once for the preview and again, the same way, for
 * the upload, and the server lays that drawing over the stored picture. The
 * two compositings round the soft edges of the letters their own way, so a
 * level or two apart is expected there and nothing more.
 */

async function dismissSetup(page: Page) {
  const setupDialog = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setupDialog).toBeVisible();
  await setupDialog.getByRole("button", { name: "Not now" }).click();
  await expect(setupDialog).toBeHidden();
}

async function gradient(page: Page): Promise<Buffer> {
  const dataUrl = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 160;
    canvas.height = 100;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no canvas");
    const fill = context.createLinearGradient(0, 0, 160, 100);
    fill.addColorStop(0, "#2d5f8b");
    fill.addColorStop(1, "#c9b37e");
    context.fillStyle = fill;
    context.fillRect(0, 0, 160, 100);
    return canvas.toDataURL("image/png");
  });
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

async function shownPixels(page: Page): Promise<number[]> {
  return page.evaluate(() => {
    const layer = document.querySelector<HTMLCanvasElement>(".studio-canvas-layers canvas:not([data-layer])");
    const context = layer?.getContext("2d");
    if (!layer || !context) return [];
    return Array.from(context.getImageData(0, 0, layer.width, layer.height).data);
  });
}

test("shows the words as adding them lays them down", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();
  await page.getByLabel("Choose an image to edit").setInputFiles({
    name: "gradient.png",
    mimeType: "image/png",
    buffer: await gradient(page),
  });

  await page.getByRole("button", { name: "Add text to the picture" }).click();
  await expect.poll(() => shownPixels(page).then((pixels) => pixels.length)).toBe(160 * 100 * 4);
  const before = await shownPixels(page);
  await page.getByRole("checkbox", { name: "Shadow" }).check();
  await page.getByRole("textbox", { name: "Words" }).fill("Harbour");
  await expect.poll(async () => (await shownPixels(page)).join() !== before.join()).toBe(true);
  const preview = await shownPixels(page);

  await page.getByRole("button", { name: "Add the words" }).click();
  await expect.poll(() => page.evaluate(async () => {
    const { id } = JSON.parse(localStorage.getItem("local-lm-studio-session") ?? "{}") as { id?: string };
    const session = (await (await fetch(`/api/studio/sessions/${id}`)).json()) as {
      messages: { role: string; parts: { type: string; metadata_json?: { provenance?: { local_edit?: { operation?: string } } } }[] }[];
    };
    const answer = session.messages.filter((message) => message.role === "assistant").at(-1);
    const record = answer?.parts.find((part) => part.type === "generation_metadata");
    return record?.metadata_json?.provenance?.local_edit?.operation ?? null;
  })).toBe("caption");
  const kept = await page.evaluate(async () => {
    const { id } = JSON.parse(localStorage.getItem("local-lm-studio-session") ?? "{}") as { id?: string };
    const session = (await (await fetch(`/api/studio/sessions/${id}`)).json()) as {
      messages: { role: string; parts: { type: string; artifact_id: string | null }[] }[];
    };
    const answer = session.messages.filter((message) => message.role === "assistant").at(-1);
    const image = answer?.parts.find((part) => part.type === "image");
    const bitmap = await createImageBitmap(await (await fetch(`/api/artifacts/${image?.artifact_id}/content`)).blob());
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d");
    if (!context) return [];
    context.drawImage(bitmap, 0, 0);
    return Array.from(context.getImageData(0, 0, bitmap.width, bitmap.height).data);
  });

  expect(kept.length).toBe(preview.length);
  const changed = before.filter((value, index) => value !== preview[index]).length;
  expect(changed).toBeGreaterThan(200);
  const worst = Math.max(...kept.map((value, index) => Math.abs(value - preview[index])));
  expect(worst).toBeLessThanOrEqual(3);
});
