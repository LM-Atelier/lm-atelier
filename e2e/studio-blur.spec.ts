import { expect, test, type Page } from "@playwright/test";

/** Blurring, pixelating or painting a marked area, through the whole path in a real browser.
 *
 * The brush marks the area on the canvas, the browser encodes and uploads it
 * as a selection, and the server works through it. What has to hold at the end
 * is the promise each tool makes: the marked part is softened, broken into
 * blocks or painted, and every pixel outside the marking is exactly what it was.
 */

async function dismissSetup(page: Page) {
  const setupDialog = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setupDialog).toBeVisible();
  await setupDialog.getByRole("button", { name: "Not now" }).click();
  await expect(setupDialog).toBeHidden();
}

/** Black and white stripes `band` pixels wide, so any blur shows as grey.
 *
 * Each case uses its own width: the same bytes are the same picture, and the
 * studio would reopen the session an earlier case left behind.
 */
async function stripes(page: Page, band: number): Promise<Buffer> {
  const dataUrl = await page.evaluate((band) => {
    const canvas = document.createElement("canvas");
    canvas.width = 96;
    canvas.height = 64;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no canvas");
    context.fillStyle = "#000000";
    context.fillRect(0, 0, 96, 64);
    context.fillStyle = "#ffffff";
    for (let x = 0; x < 96; x += band * 2) context.fillRect(x, 0, band, 64);
    return canvas.toDataURL("image/png");
  }, band);
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

/** Wait until the open session's newest step is one this edit made. */
async function waitForStep(page: Page, operation: string): Promise<void> {
  await expect.poll(() => page.evaluate(async () => {
    const { id } = JSON.parse(localStorage.getItem("local-lm-studio-session") ?? "{}") as { id?: string };
    const session = (await (await fetch(`/api/studio/sessions/${id}`)).json()) as {
      messages: { role: string; parts: { type: string; metadata_json?: { provenance?: { local_edit?: { operation?: string } } } }[] }[];
    };
    const answer = session.messages.filter((message) => message.role === "assistant").at(-1);
    const record = answer?.parts.find((part) => part.type === "generation_metadata");
    return record?.metadata_json?.provenance?.local_edit?.operation ?? null;
  })).toBe(operation);
}

/** The pixels of the newest finished picture in the open studio session. */
async function newestPicture(page: Page): Promise<{ width: number; data: number[] }> {
  return page.evaluate(async () => {
    const { id } = JSON.parse(localStorage.getItem("local-lm-studio-session") ?? "{}") as { id?: string };
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
    if (!context) return { width: 0, data: [] };
    context.drawImage(bitmap, 0, 0);
    return { width: bitmap.width, data: Array.from(context.getImageData(0, 0, bitmap.width, bitmap.height).data) };
  });
}

test("softens what the brush marked and leaves the rest exact", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();
  await page.getByLabel("Choose an image to edit").setInputFiles({
    name: "stripes.png",
    mimeType: "image/png",
    buffer: await stripes(page, 2),
  });

  await page.getByRole("button", { name: "Blur or pixelate part of the picture" }).click();
  // One dab of the default brush in the middle, from the keyboard.
  const canvas = page.getByRole("application");
  await canvas.focus();
  await canvas.press("Enter");
  await canvas.press("Enter");
  await page.getByRole("button", { name: "Blur the marked area" }).click();
  await waitForStep(page, "blur");

  const { width, data } = await newestPicture(page);
  expect(width).toBe(96);
  const red = (x: number, y: number) => data[(y * width + x) * 4];
  // The middle was marked: the stripes there have run together into grey.
  expect(red(48, 32)).toBeGreaterThan(20);
  expect(red(48, 32)).toBeLessThan(235);
  // Well outside the dab, and outside its feather, nothing moved at all.
  for (let x = 88; x < 96; x += 1) {
    for (let y = 0; y < 64; y += 1) {
      expect(red(x, y)).toBe(x % 4 < 2 ? 255 : 0);
    }
  }
});

/** A ramp whose red rises by two a column, so a block's mean is known exactly.
 *
 * A blur leaves a straight ramp as it was, so only a pixelation makes a run
 * of columns one level.
 */
async function ramp(page: Page): Promise<Buffer> {
  const dataUrl = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 96;
    canvas.height = 64;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no canvas");
    const pixels = context.createImageData(96, 64);
    for (let index = 0; index < 96 * 64; index += 1) {
      pixels.data.set([(index % 96) * 2, 60, 120, 255], index * 4);
    }
    context.putImageData(pixels, 0, 0);
    return canvas.toDataURL("image/png");
  });
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

test("pixelates what the brush marked and leaves the rest exact", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();
  await page.getByLabel("Choose an image to edit").setInputFiles({
    name: "ramp.png",
    mimeType: "image/png",
    buffer: await ramp(page),
  });

  await page.getByRole("button", { name: "Blur or pixelate part of the picture" }).click();
  await page.getByRole("button", { name: "Pixelate", exact: true }).click();
  await page.getByRole("slider", { name: "Block size" }).fill("8");
  // As for the paint: a larger brush covers the middle well inside the
  // feather and still stops short of the columns checked below.
  await page.getByRole("slider", { name: "Brush size" }).fill("120");
  const canvas = page.getByRole("application");
  await canvas.focus();
  await canvas.press("Enter");
  await canvas.press("Enter");
  await page.getByRole("button", { name: "Pixelate the marked area" }).click();
  await waitForStep(page, "pixelate");

  const { width, data } = await newestPicture(page);
  const red = (x: number, y: number) => data[(y * width + x) * 4];
  // The block from (48, 32) to (55, 39) is one level, the mean of its columns'
  // 96 to 110, where before every column differed.
  for (let x = 48; x < 56; x += 1) {
    for (let y = 32; y < 40; y += 1) {
      expect(red(x, y)).toBe(103);
    }
  }
  for (let x = 88; x < 96; x += 1) {
    for (let y = 0; y < 64; y += 1) {
      expect(red(x, y)).toBe(x * 2);
    }
  }
});

test("paints what the brush marked and leaves the rest exact", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();
  await page.getByLabel("Choose an image to edit").setInputFiles({
    name: "stripes.png",
    mimeType: "image/png",
    buffer: await stripes(page, 3),
  });

  await page.getByRole("button", { name: "Paint over part of the picture" }).click();
  await page.getByRole("button", { name: "Red", exact: true }).click();
  // The brush is sized on screen, and this small picture is shown enlarged,
  // so a larger brush is what covers its middle well inside the feather
  // while still stopping short of the columns checked below.
  await page.getByRole("slider", { name: "Brush size" }).fill("120");
  const canvas = page.getByRole("application");
  await canvas.focus();
  await canvas.press("Enter");
  await canvas.press("Enter");
  await page.getByRole("button", { name: "Paint the marked area" }).click();
  await waitForStep(page, "paint");

  const { width, data } = await newestPicture(page);
  const pixel = (x: number, y: number) => data.slice((y * width + x) * 4, (y * width + x) * 4 + 3);
  // The middle was marked, well inside the feather: it is the paint, exactly.
  expect(pixel(48, 32)).toEqual([0xe5, 0x39, 0x35]);
  for (let x = 88; x < 96; x += 1) {
    for (let y = 0; y < 64; y += 1) {
      const level = x % 6 < 3 ? 255 : 0;
      expect(pixel(x, y)).toEqual([level, level, level]);
    }
  }
});
