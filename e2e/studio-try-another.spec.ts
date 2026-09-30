import { expect, test, type Page } from "@playwright/test";

/** Try another on an Image Studio result, in a real browser.
 *
 * The runner's mock engines include an image editing workflow, so an Instruct
 * edit runs for real and its result carries the run that made it. Try another
 * reads that run and sends the same edit again on the same picture; the
 * server draws a new seed for it, and the new result joins the strip beside
 * the first, made from the original.
 */

async function dismissSetup(page: Page) {
  const setupDialog = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setupDialog).toBeVisible();
  await setupDialog.getByRole("button", { name: "Not now" }).click();
  await expect(setupDialog).toBeHidden();
}

/** A picture of its own, so it opens a studio session of its own. */
async function picture(page: Page): Promise<Buffer> {
  const dataUrl = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 96;
    canvas.height = 72;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no canvas");
    context.fillStyle = "#e9c46a";
    context.fillRect(0, 0, 96, 36);
    context.fillStyle = "#264653";
    context.fillRect(0, 36, 96, 36);
    return canvas.toDataURL("image/png");
  });
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

/** Each result's run, in the order the results were made. */
async function resultRuns(page: Page): Promise<Array<{ settings_json: { seed?: unknown }; workflow_revision_id: string | null }>> {
  return page.evaluate(async () => {
    const { id } = JSON.parse(localStorage.getItem("local-lm-studio-session") ?? "{}") as { id?: string };
    const session = (await (await fetch(`/api/studio/sessions/${id}`)).json()) as {
      messages: { role: string; parts: { type: string; metadata_json: { run_id?: string } }[] }[];
    };
    const runIds = session.messages
      .filter((message) => message.role === "assistant")
      .map((message) => message.parts.find((part) => part.type === "generation_metadata")?.metadata_json.run_id)
      .filter((runId): runId is string => Boolean(runId));
    return Promise.all(runIds.map(async (runId) => (await fetch(`/api/runs/${runId}`)).json()));
  });
}

test("makes the same edit again with a new seed, beside the first result", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();
  await page.getByLabel("Choose an image to edit").setInputFiles({
    name: "dune.png",
    mimeType: "image/png",
    buffer: await picture(page),
  });
  const instruct = page.getByRole("button", { name: "Instruct the whole image" });
  await expect(instruct).toBeEnabled();
  await instruct.click();
  await page.getByRole("textbox", { name: /Describe the edit/ }).fill("make the sand paler");
  await page.getByRole("button", { name: "Apply edit" }).click();
  const strip = page.getByRole("group", { name: "Edit history" });
  await expect(strip.getByRole("button", { name: /^Result of step 1 make the sand paler/ })).toBeVisible({ timeout: 30_000 });

  await page.getByRole("button", { name: /Try another/ }).click();

  // Made from the original, like the first, not from the first result.
  await expect(strip.getByRole("button", { name: /^Result of step 2 make the sand paler From the original/ })).toBeVisible({
    timeout: 30_000,
  });
  await expect.poll(async () => (await resultRuns(page)).length).toBe(2);
  const [first, second] = await resultRuns(page);
  expect(second.workflow_revision_id).toBe(first.workflow_revision_id);
  expect(typeof first.settings_json.seed).toBe("number");
  expect(typeof second.settings_json.seed).toBe("number");
  expect(second.settings_json.seed).not.toBe(-1);
  expect(second.settings_json.seed).not.toBe(first.settings_json.seed);
});
