import { expect, test } from "@playwright/test";

for (const width of [1280, 375]) {
  test(`recipe choices remain reachable and persist at ${width}px`, async ({ page, request }) => {
    await page.setViewportSize({ width, height: 812 });
    const session = await request.post("/api/session");
    expect(session.ok()).toBeTruthy();
    const { csrf_token } = await session.json() as { csrf_token: string };
    const headers = { "x-local-lm-csrf": csrf_token };
    const createdChat = await request.post("/api/chats", { headers, data: { title: "Recipe choices" } });
    expect(createdChat.status()).toBe(201);
    const { id: chatId } = await createdChat.json() as { id: string };
    const createdRecipe = await request.post("/api/workflow-use-case-presets", { headers, data: {
      name: `Detailed rendering ${width}`, use_case: "image_generation", settings_json: { steps: 24 },
    } });
    expect(createdRecipe.status()).toBe(201);
    const { id: recipeId } = await createdRecipe.json() as { id: string };
    await page.addInitScript((id) => localStorage.setItem("local-lm-chat", id), chatId);
    await page.goto("/");
    const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
    await expect(setup).toBeVisible();
    await setup.getByRole("button", { name: "Not now" }).click();
    const summary = page.locator("summary").filter({ hasText: "Recipes for this chat" });
    await summary.focus();
    await page.keyboard.press("Enter");
    const choices = page.getByRole("group", { name: "Use-case recipes" });
    const image = choices.getByRole("combobox", { name: "Image generation recipe" });
    await expect(image).toHaveValue("inherit");
    await expect(choices.getByRole("combobox")).toHaveCount(8);
    for (const select of await choices.getByRole("combobox").all()) {
      await expect(select).toBeEnabled();
      await select.focus();
      await expect(select).toBeFocused();
      const bounds = await select.boundingBox();
      expect(bounds).not.toBeNull();
      expect(bounds!.x).toBeGreaterThanOrEqual(0);
      expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(width);
      expect(bounds!.y).toBeGreaterThanOrEqual(0);
      expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(812);
    }
    await image.selectOption(`preset:${recipeId}`);
    await expect(image).toHaveAttribute("aria-disabled", "false");
    const endpoint = `/api/chats/${chatId}/workflow-use-case-presets/image_generation`;
    await expect.poll(async () => (await request.get(endpoint)).json()).toEqual({ mode: "preset", preset_id: recipeId });
    await image.focus();
    await page.keyboard.press("Home");
    await expect.poll(async () => (await request.get(endpoint)).json()).toEqual({ mode: "inherit" });
    await expect(image).toHaveValue("inherit");
    await expect(image).toHaveAttribute("aria-disabled", "false");
    await expect(image).toBeFocused();
    await page.keyboard.press("ArrowDown");
    await expect.poll(async () => (await request.get(endpoint)).json()).toEqual({ mode: "automatic" });
    await expect(image).toBeFocused();
    await page.reload();
    await summary.focus();
    await page.keyboard.press("Enter");
    await expect(image).toHaveValue("automatic");
    await expect(choices.getByRole("combobox", { name: "Whole-image edit recipe" })).toHaveValue("inherit");
    await expect(page.getByRole("combobox", { name: "Generation mode" })).toHaveValue("auto");
    await image.focus();
    for (const control of [summary, page.getByRole("textbox", { name: "Message" }), page.getByRole("combobox", { name: "Generation mode" })]) {
      const bounds = await control.boundingBox();
      expect(bounds).not.toBeNull();
      expect(bounds!.y).toBeGreaterThanOrEqual(0);
      expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(812);
    }
    await page.screenshot({ path: `test-results/recipe-choices-${width}.png` });
    await summary.focus();
    await page.keyboard.press("Enter");
    const message = page.getByRole("textbox", { name: "Message" });
    await message.focus();
    await expect(message).toBeFocused();
  });
}
