import { expect, test, type Page } from "@playwright/test";

/** A Studio edit that is waiting shows where it stands, and stopping it ends it, in a real browser.
 *
 * The runner's mock engines finish an edit in a fraction of a second, so the
 * generation queue is paused first, through the same request the queue panel
 * sends: the edit is then accepted and waits, which is when a person reads its
 * progress and may decide to stop it.
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
    return canvas.toDataURL("image/png");
  });
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

/** Pause or resume the generation lane as the queue panel does. */
async function generation(page: Page, action: "pause-after-current" | "resume") {
  const status = await page.evaluate(async (step) => {
    const { csrf_token: token } = (await (await fetch("/api/session", { method: "POST" })).json()) as { csrf_token: string };
    const lane = (await (await fetch("/api/queue/lanes/generation")).json()) as { revision: number };
    const response = await fetch(`/api/queue/lanes/generation/${step}`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-local-lm-csrf": token },
      body: JSON.stringify({ expected_revision: lane.revision, idempotency_key: crypto.randomUUID() }),
    });
    return response.status;
  }, action);
  expect(status).toBeLessThan(300);
}

test("shows a waiting edit's progress and stops it, leaving the picture as it was", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();
  await page.getByLabel("Choose an image to edit").setInputFiles({
    name: "sky.png",
    mimeType: "image/png",
    buffer: await picture(page),
  });
  const instruct = page.getByRole("button", { name: "Instruct the whole image" });
  await expect(instruct).toBeEnabled();
  await instruct.click();
  await generation(page, "pause-after-current");

  try {
    await page.getByRole("textbox", { name: /Describe the edit/ }).fill("make the sky warmer");
    await page.getByRole("button", { name: "Apply edit" }).click();

    const stop = page.getByRole("button", { name: "Stop the edit" });
    await expect(stop).toBeVisible({ timeout: 15_000 });
    // The step it has reached, beside the press that stops it.
    const progress = page.locator(".studio-panel .generation-progress");
    await expect(progress).toBeVisible();
    await expect(progress).toHaveText(/\S/);
    await stop.click();

    // Stopped: the progress goes, Apply is offered again, and no result joined the history.
    await expect(stop).toBeHidden({ timeout: 15_000 });
    await expect(progress).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Apply edit" })).toBeVisible();
    await expect(page.getByRole("group", { name: "Edit history" }).getByRole("button")).toHaveCount(1);
  } finally {
    await generation(page, "resume");
  }
});
