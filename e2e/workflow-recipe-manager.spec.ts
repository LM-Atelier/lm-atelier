import { expect, test } from "@playwright/test";

for (const width of [1280, 375]) {
  test(`manage recipe settings and scope choices at ${width}px`, async ({ page, request }) => {
    await page.setViewportSize({ width, height: 812 });
    const session = await request.post("/api/session");
    expect(session.ok()).toBeTruthy();
    const { csrf_token } = await session.json() as { csrf_token: string };
    const headers = { "x-local-lm-csrf": csrf_token };
    const workflowResponse = await request.post("/api/workflows", { headers, data: {
      name: `Reference ${width}`, operation: "text_to_image", description: "Neutral setting controls",
      engine: "mock", api_graph: {}, ui_graph: {}, dependencies: {}, input_schema: { properties: {
        recipe_quality: { type: "number", title: "Recipe quality", default: 0.5, minimum: 0, maximum: 1 },
      } },
    } });
    expect(workflowResponse.status()).toBe(201);
    const workflow = await workflowResponse.json() as { id: string };
    const beforeWorkflows = await (await request.get("/api/workflow-summaries")).json() as unknown[];
    const projectResponse = await request.post("/api/projects", { headers, data: { name: `Recipe project ${width}` } });
    expect(projectResponse.status()).toBe(201);
    const project = await projectResponse.json() as { id: string };
    await page.goto("/?view=workflows");
    const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
    await expect(setup).toBeVisible();
    await setup.getByRole("button", { name: "Not now" }).click();
    await page.getByRole("button", { name: "Manage recipes" }).click();
    const dialog = page.getByRole("dialog", { name: "Workflow recipes" });
    await dialog.getByRole("button", { name: "New recipe" }).click();
    const recipeName = `Recipe ${width}`;
    await dialog.getByRole("textbox", { name: "Recipe name" }).fill(recipeName);
    await dialog.getByRole("combobox", { name: "Reference workflow for settings" }).selectOption(workflow.id);
    await dialog.getByRole("checkbox", { name: "Include Recipe quality" }).check();
    await dialog.getByRole("spinbutton", { name: "Recipe quality" }).fill("0.75");
    const editorFits = await dialog.evaluate((element) => element.scrollWidth <= element.clientWidth);
    expect(editorFits).toBe(true);
    const bounds = await dialog.boundingBox();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(width);
    await page.screenshot({ path: `test-results/recipe-editor-${width}.png` });
    await dialog.getByRole("button", { name: "Save recipe" }).click();
    await expect(dialog.getByRole("button", { name: `Edit recipe ${recipeName}` })).toBeVisible();
    const recipes = await (await request.get("/api/workflow-use-case-presets?limit=200")).json() as Array<{ id: string; name: string; settings_json: Record<string, number> }>;
    const recipe = recipes.find((item) => item.name === recipeName)!;
    expect(recipe.settings_json).toEqual({ recipe_quality: 0.75 });
    expect((await (await request.get("/api/workflow-summaries")).json() as unknown[]).length).toBe(beforeWorkflows.length);
    await dialog.locator("summary").filter({ hasText: "Workspace defaults" }).click();
    const defaults = dialog.locator("details").filter({ has: page.locator("summary", { hasText: "Workspace defaults" }) });
    await defaults.getByRole("combobox", { name: "Image generation recipe" }).selectOption(`preset:${recipe.id}`);
    await expect.poll(async () => (await request.get("/api/workflow-use-case-defaults/image_generation")).json()).toEqual({ preset_id: recipe.id });
    await dialog.locator("summary").filter({ hasText: "Project choices" }).click();
    const projects = dialog.locator("details").filter({ has: page.locator("summary", { hasText: "Project choices" }) });
    await projects.getByRole("combobox", { name: "Project for recipe choices" }).selectOption(project.id);
    const projectChoice = projects.getByRole("combobox", { name: "Image generation recipe" });
    await expect(projectChoice).toHaveValue("inherit");
    await projectChoice.selectOption(`preset:${recipe.id}`);
    const choiceUrl = `/api/projects/${project.id}/workflow-use-case-presets/image_generation`;
    await expect.poll(async () => (await request.get(choiceUrl)).json()).toEqual({ mode: "preset", preset_id: recipe.id });
    await dialog.getByRole("button", { name: `Edit recipe ${recipeName}` }).click();
    await dialog.getByRole("textbox", { name: "Recipe name" }).fill(`${recipeName} revised`);
    await dialog.getByRole("button", { name: "Save recipe" }).click();
    await expect(dialog.getByRole("button", { name: `Edit recipe ${recipeName} revised` })).toBeVisible();
    const saved = await (await request.get(`/api/workflow-use-case-presets/${recipe.id}`)).json() as { settings_json: Record<string, number> };
    expect(saved.settings_json).toEqual({ recipe_quality: 0.75 });
    await dialog.getByRole("button", { name: `Delete recipe ${recipeName} revised` }).click();
    await dialog.getByRole("button", { name: "Confirm deletion" }).click();
    await expect(dialog.getByText("Remove this recipe's selections before disabling or deleting it.")).toBeVisible();
    await dialog.getByRole("button", { name: "Keep recipe" }).click();
    await defaults.getByRole("combobox", { name: "Image generation recipe" }).selectOption("automatic");
    await expect.poll(async () => (await request.get("/api/workflow-use-case-defaults/image_generation")).json()).toEqual({ preset_id: null });
    await projectChoice.selectOption("inherit");
    await expect.poll(async () => (await request.get(choiceUrl)).json()).toEqual({ mode: "inherit" });
    await dialog.getByRole("button", { name: `Delete recipe ${recipeName} revised` }).click();
    await dialog.getByRole("button", { name: "Confirm deletion" }).click();
    await expect(dialog.getByText("Recipe deleted.")).toBeVisible();
    expect((await request.get(`/api/workflow-use-case-presets/${recipe.id}`)).status()).toBe(404);
    await page.keyboard.press("Escape");
    await expect(dialog).toBeHidden();
    await expect(page.getByRole("button", { name: "Manage recipes" })).toBeFocused();
  });
}
