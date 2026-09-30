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

async function openWords(page: Page) {
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
}

async function shownPixels(page: Page): Promise<number[]> {
  return page.evaluate(() => {
    const layer = document.querySelector<HTMLCanvasElement>(".studio-canvas-layers canvas:not([data-layer])");
    const context = layer?.getContext("2d");
    if (!layer || !context) return [];
    return Array.from(context.getImageData(0, 0, layer.width, layer.height).data);
  });
}

/** The picture on the canvas once it has changed from `was` and stopped changing.
 *
 * The preview follows each step of a drag and each move of a slider, so the
 * first picture that differs can be one it has already moved on from.
 */
async function settledPixels(page: Page, was: number[]): Promise<number[]> {
  const before = was.join();
  let last = "";
  let pixels: number[] = [];
  await expect.poll(async () => {
    pixels = await shownPixels(page);
    const now = pixels.join();
    const settled = now !== before && now === last;
    last = now;
    return settled;
  }, { intervals: [150] }).toBe(true);
  return pixels;
}

/** How many answers the Studio's session holds. The same picture opens the same
 * session, so a case can find another case's steps already there. */
async function answerCount(page: Page): Promise<number> {
  return page.evaluate(async () => {
    const { id } = JSON.parse(localStorage.getItem("local-lm-studio-session") ?? "{}") as { id?: string };
    if (!id) return 0;
    const session = (await (await fetch(`/api/studio/sessions/${id}`)).json()) as { messages: { role: string }[] };
    return session.messages.filter((message) => message.role === "assistant").length;
  });
}

/** The pixels of the picture the next Studio step made, once there is a step
 * past the `answers` there were and it is the added words. */
async function addedPixels(page: Page, answers: number): Promise<number[]> {
  await expect.poll(() => page.evaluate(async (earlier) => {
    const { id } = JSON.parse(localStorage.getItem("local-lm-studio-session") ?? "{}") as { id?: string };
    const session = (await (await fetch(`/api/studio/sessions/${id}`)).json()) as {
      messages: { role: string; parts: { type: string; metadata_json?: { provenance?: { local_edit?: { operation?: string } } } }[] }[];
    };
    const replies = session.messages.filter((message) => message.role === "assistant");
    if (replies.length <= earlier) return null;
    const record = replies.at(-1)?.parts.find((part) => part.type === "generation_metadata");
    return record?.metadata_json?.provenance?.local_edit?.operation ?? null;
  }, answers)).toBe("caption");
  return page.evaluate(async () => {
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
}

/** Where the words' ink lies: the box and the middle of every pixel that differs from the picture without them. */
function ink(pixels: number[], picture: number[]) {
  let [count, sumX, sumY] = [0, 0, 0];
  let [left, top, right, bottom] = [Infinity, Infinity, -Infinity, -Infinity];
  for (let at = 0; at < pixels.length; at += 4) {
    if (pixels[at] === picture[at] && pixels[at + 1] === picture[at + 1] && pixels[at + 2] === picture[at + 2]) continue;
    const x = (at / 4) % 160;
    const y = Math.floor(at / 4 / 160);
    [count, sumX, sumY] = [count + 1, sumX + x, sumY + y];
    [left, top, right, bottom] = [Math.min(left, x), Math.min(top, y), Math.max(right, x), Math.max(bottom, y)];
  }
  return { middle: { x: sumX / count, y: sumY / count }, width: right - left + 1, height: bottom - top + 1 };
}

test("shows the words as adding them lays them down", async ({ page }) => {
  await openWords(page);
  const before = await shownPixels(page);
  await page.getByRole("checkbox", { name: "Shadow" }).check();
  await page.getByRole("textbox", { name: "Words" }).fill("Harbour");
  const preview = await settledPixels(page, before);

  const answers = await answerCount(page);
  await page.getByRole("button", { name: "Add the words" }).click();
  const kept = await addedPixels(page, answers);

  expect(kept.length).toBe(preview.length);
  const changed = before.filter((value, index) => value !== preview[index]).length;
  expect(changed).toBeGreaterThan(200);
  const worst = Math.max(...kept.map((value, index) => Math.abs(value - preview[index])));
  expect(worst).toBeLessThanOrEqual(3);
});

test("moves the words as far as they are dragged, turns them, and lays them down as shown", async ({ page }) => {
  await openWords(page);
  const before = await shownPixels(page);
  await page.getByRole("textbox", { name: "Words" }).fill("Harbour");
  const placed = await settledPixels(page, before);

  // Dragged 20 of the picture's pixels left and 30 up, from wherever the press lands.
  const picture = await page.locator(".studio-canvas-layers canvas:not([data-layer])").boundingBox();
  if (!picture) throw new Error("the picture is not on screen");
  const scale = picture.width / 160;
  const from = { x: picture.x + picture.width / 2, y: picture.y + picture.height * 0.8 };
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(from.x - 20 * scale, from.y - 30 * scale, { steps: 6 });
  await page.mouse.up();
  const moved = await settledPixels(page, placed);

  const start = ink(placed, before);
  const end = ink(moved, before);
  expect(Math.abs(end.middle.x - start.middle.x + 20)).toBeLessThanOrEqual(1);
  expect(Math.abs(end.middle.y - start.middle.y + 30)).toBeLessThanOrEqual(1);
  expect(start.width).toBeGreaterThan(start.height);

  // A quarter turn stands the line of words on its end where it is. The ink's
  // middle is a pixel or two from the middle of the block the words turn about,
  // so it moves that much; turning about anything else would carry the words
  // tens of pixels away.
  await page.getByRole("slider", { name: "Turn" }).fill("90");
  const turned = await settledPixels(page, moved);
  const upright = ink(turned, before);
  expect(upright.height).toBeGreaterThan(upright.width);
  expect(Math.hypot(upright.middle.x - end.middle.x, upright.middle.y - end.middle.y)).toBeLessThanOrEqual(5);

  const answers = await answerCount(page);
  await page.getByRole("button", { name: "Add the words" }).click();
  const kept = await addedPixels(page, answers);

  expect(kept.length).toBe(turned.length);
  const worst = Math.max(...kept.map((value, index) => Math.abs(value - turned[index])));
  expect(worst).toBeLessThanOrEqual(3);
});
