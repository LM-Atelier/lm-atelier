import { expect, test, type Page } from "@playwright/test";

/** Several results from one Image Studio edit, in a real browser.
 *
 * The runner's mock engines include an image editing workflow, so an Instruct
 * edit runs for real. Asked for three results, the edit comes back as three
 * steps made from the same picture, each from a run of its own with its own
 * seed, so they sit beside one another as alternatives.
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
    context.fillStyle = "#2a9d8f";
    context.fillRect(0, 0, 96, 36);
    context.fillStyle = "#f4a261";
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

test("asks one edit for three results and shows each beside the others", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();
  await page.getByLabel("Choose an image to edit").setInputFiles({
    name: "harbour.png",
    mimeType: "image/png",
    buffer: await picture(page),
  });
  const instruct = page.getByRole("button", { name: "Instruct the whole image" });
  await expect(instruct).toBeEnabled();
  await instruct.click();
  await page.getByRole("textbox", { name: /Describe the edit/ }).fill("make the water calmer");
  await page.getByRole("group", { name: "Results" }).getByRole("button", { name: "3" }).click();
  await page.getByRole("button", { name: "Apply edit" }).click();

  const strip = page.getByRole("group", { name: "Edit history" });
  await expect(strip.getByRole("button", { name: /^Result of step 1 make the water calmer/ })).toBeVisible({ timeout: 30_000 });
  // The second and third were made from the original too, not from the result before them.
  for (const step of [2, 3]) {
    await expect(
      strip.getByRole("button", { name: new RegExp(`^Result of step ${step} make the water calmer From the original`) }),
    ).toBeVisible({ timeout: 30_000 });
  }
  await expect.poll(async () => (await resultRuns(page)).length).toBe(3);
  const runs = await resultRuns(page);
  expect(new Set(runs.map((run) => run.workflow_revision_id)).size).toBe(1);
  const seeds = runs.map((run) => run.settings_json.seed);
  for (const seed of seeds) expect(typeof seed).toBe("number");
  expect(new Set(seeds).size).toBe(3);
});
