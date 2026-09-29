import { expect, test, type Page } from "@playwright/test";

/** Keeping an Image Studio result as a recipe and using it again, in a real browser.
 *
 * The runner's mock engines include an image editing workflow, so an Instruct
 * edit runs for real and its result carries the run that made it. Saving that
 * result as a recipe reads the recipe from the run on the server; choosing the
 * recipe afterwards brings back its words and the workflow the run used.
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
    context.fillRect(0, 0, 120, 40);
    context.fillStyle = "#4f7942";
    context.fillRect(0, 40, 120, 40);
    return canvas.toDataURL("image/png");
  });
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

test("saves a model's result as a recipe from its run, and applies the recipe again", async ({ page }) => {
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

  // An edit a model makes, so its result has a run to save a recipe from.
  await page.getByRole("textbox", { name: /Describe the edit/ }).fill("make the sky warmer");
  await page.getByRole("button", { name: "Apply edit" }).click();
  const strip = page.getByRole("group", { name: "Edit history" });
  await expect(strip.getByRole("button", { name: /^Result of step 1 make the sky warmer/ })).toBeVisible({ timeout: 30_000 });

  const name = page.getByRole("textbox", { name: "Save this edit as a recipe" });
  await expect(name).toBeVisible();
  await name.fill("Warmer sky");
  await page.getByRole("button", { name: "Save recipe" }).click();
  await expect(page.getByText("Saved.")).toBeVisible();
  const saved = page.locator(".studio-recipes").getByRole("button", { name: "Warmer sky" });
  await expect(saved).toBeVisible();

  // Read on the server from what the run used, and nothing of the selection.
  const recipes = (await page.evaluate(async () => (await fetch("/api/edit-templates")).json())) as Array<{
    name: string;
    instruction: string;
    workflow_revision_id: string | null;
    mask_mode: string;
  }>;
  const recipe = recipes.find((entry) => entry.name === "Warmer sky");
  expect(recipe?.instruction).toBe("make the sky warmer");
  expect(recipe?.workflow_revision_id).toBeTruthy();
  expect(recipe?.mask_mode).toBe("none");

  // Choosing it again brings back its words and its workflow, and it runs.
  await saved.click();
  await expect(page.getByRole("textbox", { name: /Describe the edit/ })).toHaveValue("make the sky warmer");
  await expect(page.getByText("Warmer sky supplies the workflow for this edit.")).toBeVisible();
  await page.getByRole("button", { name: "Apply edit" }).click();
  await expect(strip.getByRole("button", { name: /^Result of step 2 make the sky warmer/ })).toBeVisible({ timeout: 30_000 });
});
