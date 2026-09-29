import { expect, test, type Page } from "@playwright/test";

/** Going back in Image Studio's history and changing an earlier picture, in a real browser.
 *
 * The history strip lists results in the order they were made. A change made
 * from an earlier picture is listed last, so it has to say where it came from
 * or it reads as the next change to the latest picture. Exact edits need no
 * model, so this branches with flips and a turn.
 */

async function dismissSetup(page: Page) {
  const setupDialog = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setupDialog).toBeVisible();
  await setupDialog.getByRole("button", { name: "Not now" }).click();
  await expect(setupDialog).toBeHidden();
}

async function picture(page: Page): Promise<Buffer> {
  const dataUrl = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 120;
    canvas.height = 80;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no canvas");
    context.fillStyle = "#8fb8de";
    context.fillRect(0, 0, 120, 80);
    context.fillStyle = "#4f7942";
    context.fillRect(0, 40, 60, 40);
    return canvas.toDataURL("image/png");
  });
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

test("names the picture a change was made from when it was not the latest", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();
  await page.getByLabel("Choose an image to edit").setInputFiles({
    name: "blocks.png",
    mimeType: "image/png",
    buffer: await picture(page),
  });
  const strip = page.getByRole("group", { name: "Edit history" });
  const turn = page.getByRole("button", { name: "Rotate, straighten or flip" });
  await expect(turn).toBeEnabled();
  await turn.click();

  await page.getByRole("button", { name: "Flip horizontally" }).click();
  await expect(strip.getByRole("button", { name: /^Result of step 1 Flip horizontally$/ })).toBeVisible();
  await page.getByRole("button", { name: "Flip vertically" }).click();
  await expect(strip.getByRole("button", { name: /^Result of step 2 Flip vertically$/ })).toBeVisible();

  // Back to the original, and a change from there.
  await strip.getByRole("button", { name: /The original image/ }).click();
  await page.getByRole("button", { name: "Rotate right" }).click();

  await expect(strip.getByRole("button", { name: /^Result of step 3 Rotate right From the original$/ })).toBeVisible();
  // The changes made one after another still say nothing more.
  await expect(strip.getByRole("button", { name: /^Result of step 2 Flip vertically$/ })).toBeVisible();

  // Back to the first flip, and a change from there.
  await strip.getByRole("button", { name: /^Result of step 1 / }).click();
  await page.getByRole("button", { name: "Rotate left" }).click();
  await expect(strip.getByRole("button", { name: /^Result of step 4 Rotate left From step 1$/ })).toBeVisible();
});
