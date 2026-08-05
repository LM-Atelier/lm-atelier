import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

const PROJECT_NAME = "Golden Path Studio";
const MODEL_NAME = "Tiny Safe Fixture";
const STORY_PROMPT = "Write a two-sentence story about an orchard robot named Pip.";
const IMAGE_PROMPT = "Make an image based on the previous story, with Pip under an apple tree.";

async function dismissSetup(page: Page) {
  const setupDialog = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setupDialog).toBeVisible();
  await setupDialog.getByRole("button", { name: "Not now" }).click();
  await expect(setupDialog).toBeHidden();
}

async function createSession(request: APIRequestContext): Promise<string> {
  const response = await request.post("/api/session");
  expect(response.ok()).toBeTruthy();
  const payload = await response.json() as { csrf_token: string };
  return payload.csrf_token;
}

async function registerSafeFixtureModel(request: APIRequestContext, modelPath: string) {
  const csrfToken = await createSession(request);
  const response = await request.post("/api/models/import", {
    headers: { "x-local-lm-csrf": csrfToken },
    data: {
      name: MODEL_NAME,
      role: "chat",
      engine: "mock",
      local_path: modelPath,
    },
  });
  expect(response.status()).toBe(201);
  const model = await response.json() as {
    name: string;
    compatibility: string;
    manifest_json: Record<string, unknown>;
  };
  expect(model).toMatchObject({
    name: MODEL_NAME,
    compatibility: "advanced_import",
    manifest_json: {
      imported: true,
      pickle_compatible_weights: false,
    },
  });

  const modelsResponse = await request.get("/api/models");
  expect(modelsResponse.ok()).toBeTruthy();
  const models = await modelsResponse.json() as Array<{
    name: string;
    active: boolean;
  }>;
  expect(models).toEqual(expect.arrayContaining([
    expect.objectContaining({ name: MODEL_NAME, active: true }),
  ]));

  const profilesResponse = await request.get("/api/profiles");
  expect(profilesResponse.ok()).toBeTruthy();
  const profiles = await profilesResponse.json() as Array<{
    name: string;
    role: string;
    model_install_id: string | null;
  }>;
  expect(profiles).toEqual(expect.arrayContaining([
    expect.objectContaining({
      name: MODEL_NAME,
      role: "chat",
      model_install_id: expect.any(String),
    }),
  ]));
}

async function expectPersistedConversation(page: Page) {
  await expect(page.getByRole("heading", { name: STORY_PROMPT })).toBeVisible();
  const messages = page.locator(".messages > article.message");
  await expect(messages).toHaveCount(4);

  const history = await messages.evaluateAll((elements) => elements.map((element) => ({
    role: element.classList.contains("user") ? "user" : "assistant",
    text: element.textContent ?? "",
    hasImage: Boolean(element.querySelector('img[alt="Generated image"]')),
  })));
  expect(history.map((entry) => entry.role)).toEqual([
    "user",
    "assistant",
    "user",
    "assistant",
  ]);
  expect(history[0].text).toContain(STORY_PROMPT);
  expect(history[1].text).toContain("Mock local response");
  expect(history[1].text).toContain(STORY_PROMPT);
  expect(history[2].text).toContain(IMAGE_PROMPT);
  expect(history[3].hasImage).toBe(true);
}

test("persists a streamed text and contextual image golden path", async ({
  browser,
  page,
  request,
}) => {
  const modelPath = process.env.LM_ATELIER_E2E_MODEL_PATH;
  expect(modelPath, "the isolated runner must provide a model fixture").toBeTruthy();
  await registerSafeFixtureModel(request, modelPath!);

  const eventTypes: string[] = [];
  page.on("websocket", (socket) => {
    socket.on("framereceived", ({ payload }) => {
      if (typeof payload !== "string") return;
      try {
        const event = JSON.parse(payload) as { type?: string };
        if (event.type) eventTypes.push(event.type);
      } catch {
        // Ignore non-JSON websocket control frames.
      }
    });
  });

  await page.goto("/");
  await expect(page.locator(".brand > span")).toContainText("LM Atelier");
  await dismissSetup(page);
  const previewPath = process.env.LM_ATELIER_E2E_PREVIEW_PATH?.trim();
  if (previewPath) {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.screenshot({
      path: previewPath,
      animations: "disabled",
    });
  }

  // Tab resumes from the sequential focus navigation starting point, which a
  // blur does not reset: after the setup dialog closes, that point is wherever
  // the dialog left it, and Tab walks forward from there. The assertion is
  // that the first tab stop is the skip link, so the walk has to start at the
  // top of the document.
  //
  // Focusing the root element is what moves it. It passed for a long time
  // because the elements after that point happened to be exhausted, so Tab
  // wrapped around to the beginning - adding one control anywhere later was
  // enough to stop the wrap and land on it instead.
  await page.evaluate(() => {
    (document.activeElement as HTMLElement | null)?.blur();
    document.documentElement.focus();
  });
  await page.keyboard.press("Tab");
  // Reported rather than assumed: two plausible explanations for this failing
  // both turned out to be wrong, and the cheapest way to stop guessing is to
  // make the failure say what actually holds focus.
  const focused = await page.evaluate(() => {
    const active = document.activeElement as HTMLElement | null;
    if (!active) return "nothing";
    const label = active.getAttribute("aria-label") ?? active.textContent?.trim().slice(0, 40);
    return `${active.tagName}.${active.className} [${label ?? ""}]`;
  });
  expect(focused, "the first tab stop should be the skip link").toContain("skip-link");
  await page.keyboard.press("Enter");
  await expect(page.locator("#main-content")).toBeFocused();

  const newProject = page.getByRole("button", { name: "New project" });
  await expect(newProject).toHaveAccessibleName("New project");
  await newProject.focus();
  await page.keyboard.press("Enter");
  // Naming a project happens in the app now, so the whole exchange stays
  // reachable from the keyboard instead of handing off to OS chrome.
  await page.getByRole("textbox", { name: "Project name" }).fill(PROJECT_NAME);
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByText(PROJECT_NAME, { exact: true })).toBeVisible();

  const newChat = page.getByRole("button", { name: `New chat in ${PROJECT_NAME}` });
  await expect(newChat).toHaveAccessibleName(`New chat in ${PROJECT_NAME}`);
  await newChat.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { name: "New chat" })).toBeVisible();

  const composer = page.getByRole("textbox", { name: "Message" });
  const mode = page.getByRole("combobox", { name: "Generation mode" });
  await mode.selectOption("text");
  await composer.fill(STORY_PROMPT);
  await composer.press("Enter");

  await expect.poll(
    () => eventTypes.filter((eventType) => eventType === "text.delta").length,
    { message: "the browser should receive multiple incremental text frames" },
  ).toBeGreaterThan(1);
  await expect(page.locator(".message.assistant").filter({
    hasText: "Mock local response",
  })).toContainText(STORY_PROMPT);
  await expect(page.getByRole("button", { name: "Regenerate response" })).toBeVisible();

  await mode.selectOption("image");
  await expect(mode).toHaveValue("image");
  await composer.fill(IMAGE_PROMPT);
  await composer.press("Enter");
  const generatedImage = page.getByRole("img", { name: "Generated image" });
  await expect(generatedImage).toBeVisible();
  await expect.poll(
    () => generatedImage.evaluate((image) => (image as HTMLImageElement).naturalWidth),
    { message: "the generated image should load through the authenticated artifact route" },
  ).toBeGreaterThan(0);

  const chatsResponse = await request.get("/api/chats?include_archived=true&query=");
  expect(chatsResponse.ok()).toBeTruthy();
  const chats = await chatsResponse.json() as Array<{ id: string }>;
  expect(chats).toHaveLength(1);
  const chatResponse = await request.get(`/api/chats/${chats[0].id}`);
  expect(chatResponse.ok()).toBeTruthy();
  const persistedChat = await chatResponse.json() as {
    messages: Array<{
      role: string;
      parts: Array<{
        type: string;
        metadata_json: {
          provenance?: {
            routing?: {
              standalone_prompt?: string;
              text_context?: string;
            };
            visual_prompt?: {
              applied?: boolean;
              original_prompt?: string;
            };
          };
        };
      }>;
    }>;
  };
  const imageAssistant = persistedChat.messages.filter(
    (message) => message.role === "assistant",
  ).at(-1);
  const generationMetadata = imageAssistant?.parts.find(
    (part) => part.type === "generation_metadata",
  );
  const provenance = generationMetadata?.metadata_json.provenance;
  const contextualPrompt = provenance?.routing?.standalone_prompt;
  const compiledVisualPrompt = provenance?.visual_prompt;
  // The link to the earlier text still has to survive persistence, but it now
  // survives as one compiled description plus the passage it was compiled from,
  // rather than as the request with a chat excerpt pasted underneath it.
  expect(compiledVisualPrompt?.applied).toBe(true);
  expect(contextualPrompt).not.toContain("Source chat text:");
  expect(provenance?.routing?.text_context).toBeTruthy();
  expect(compiledVisualPrompt?.original_prompt).toContain("Source chat text:");
  expect(compiledVisualPrompt?.original_prompt).toContain(STORY_PROMPT);

  await page.reload();
  await expectPersistedConversation(page);

  const restartedContext = await browser.newContext();
  try {
    const restartedPage = await restartedContext.newPage();
    await restartedPage.goto("/");
    await dismissSetup(restartedPage);
    await expectPersistedConversation(restartedPage);
  } finally {
    await restartedContext.close();
  }

  const mobileContext = await browser.newContext({
    viewport: { width: 375, height: 667 },
  });
  try {
    const mobilePage = await mobileContext.newPage();
    await mobilePage.goto("/");
    await dismissSetup(mobilePage);
    await expectPersistedConversation(mobilePage);
    await expect(mobilePage.getByRole("textbox", { name: "Message" })).toBeVisible();
    await expect(mobilePage.getByRole("combobox", { name: "Generation mode" })).toBeVisible();
    const mainFitsViewport = await mobilePage.locator("#main-content").evaluate(
      (element) => element.scrollWidth <= element.clientWidth,
    );
    expect(mainFitsViewport, "the mobile chat header must not overflow the main viewport").toBe(true);

    const menu = mobilePage.getByRole("button", { name: "Toggle navigation" });
    await menu.click();
    await expect(menu).toHaveAttribute("aria-expanded", "true");
    const mediaLibrary = mobilePage.getByRole("button", { name: "Media library" });
    await mediaLibrary.focus();
    await mobilePage.keyboard.press("Enter");
    await expect(menu).toHaveAttribute("aria-expanded", "false");
    await expect(mobilePage.locator("#main-content")).toBeFocused();
    await menu.click();
    await expect(
      mobilePage.getByRole("button", { name: "Media library" }),
    ).toHaveAttribute("aria-current", "page");
  } finally {
    await mobileContext.close();
  }
});
