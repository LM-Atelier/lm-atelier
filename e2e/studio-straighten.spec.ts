import { expect, test, type Page } from "@playwright/test";

/** Straightening in a real browser: the canvas shows the turn, and the press keeps its box.
 *
 * The browser draws the turned picture itself while the slider moves, and the
 * server makes the kept picture with its own resampling, so the two are not
 * compared pixel for pixel. What has to hold is that the canvas changes as the
 * slider moves and that the kept picture is the box both sides work out.
 */

async function dismissSetup(page: Page) {
  const setupDialog = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setupDialog).toBeVisible();
  await setupDialog.getByRole("button", { name: "Not now" }).click();
  await expect(setupDialog).toBeHidden();
}

async function horizon(page: Page): Promise<Buffer> {
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

async function shownPixels(page: Page): Promise<number[]> {
  return page.evaluate(() => {
    const layer = document.querySelector<HTMLCanvasElement>(".studio-canvas-layers canvas:not([data-layer])");
    const context = layer?.getContext("2d");
    if (!layer || !context) return [];
    return Array.from(context.getImageData(0, 0, layer.width, layer.height).data);
  });
}

test("shows the turn as the slider moves and keeps the box it covers", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();
  await page.getByLabel("Choose an image to edit").setInputFiles({
    name: "horizon.png",
    mimeType: "image/png",
    buffer: await horizon(page),
  });

  await page.getByRole("button", { name: "Rotate, straighten or flip" }).click();
  await expect.poll(() => shownPixels(page).then((pixels) => pixels.length)).toBe(160 * 100 * 4);
  const level = await shownPixels(page);
  await page.getByRole("slider", { name: "Straighten" }).fill("10");
  await expect(page.getByText("10° clockwise")).toBeVisible();
  await expect.poll(async () => (await shownPixels(page)).join() !== level.join()).toBe(true);

  await page.getByRole("button", { name: "Straighten", exact: true }).click();
  await expect.poll(() => page.evaluate(async () => {
    const { id } = JSON.parse(localStorage.getItem("local-lm-studio-session") ?? "{}") as { id?: string };
    const session = (await (await fetch(`/api/studio/sessions/${id}`)).json()) as {
      messages: { role: string; parts: { type: string; artifact_id: string | null; metadata_json?: { provenance?: { local_edit?: { operation?: string } } } }[] }[];
    };
    const answer = session.messages.filter((message) => message.role === "assistant").at(-1);
    if (answer?.parts.find((part) => part.type === "generation_metadata")?.metadata_json?.provenance?.local_edit?.operation !== "straighten") return null;
    const image = answer.parts.find((part) => part.type === "image");
    const bitmap = await createImageBitmap(await (await fetch(`/api/artifacts/${image?.artifact_id}/content`)).blob());
    return `${bitmap.width}x${bitmap.height}`;
  })).toBe("120x75");
});
