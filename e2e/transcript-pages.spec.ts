import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

async function session(request: APIRequestContext) {
  const response = await request.post("/api/session");
  expect(response.ok()).toBeTruthy();
  return { "x-local-lm-csrf": (await response.json() as { csrf_token: string }).csrf_token };
}

test.beforeAll(async ({ request }) => {
  const headers = await session(request);
  const modelPath = process.env.LM_ATELIER_E2E_MODEL_PATH;
  expect(modelPath).toBeTruthy();
  const imported = await request.post("/api/models/import", { headers,
    data: { name: "Transcript fixture", role: "chat", engine: "mock", local_path: modelPath } });
  expect(imported.status()).toBe(201);
});

async function exerciseTranscriptActions(page: Page, id: string) {
  const assistant = page.locator(".messages > .message.assistant").last();
  await assistant.getByRole("button", { name: "Good response", exact: true }).focus();
  await assistant.getByRole("button", { name: "Good response", exact: true }).press("Enter");
  await expect(assistant.getByRole("button", { name: "Good response", exact: true })).toHaveAttribute("aria-pressed", "true");
  await assistant.getByRole("button", { name: "Regenerate response", exact: true }).focus();
  await assistant.getByRole("button", { name: "Regenerate response", exact: true }).press("Enter");
  await expect(assistant.getByRole("button", { name: "Previous response revision" })).toBeEnabled();
  await assistant.getByRole("button", { name: "Previous response revision" }).click();
  await expect(assistant.getByText("1 / 2", { exact: true })).toBeVisible();
  await assistant.getByRole("button", { name: "Next response revision" }).click();
  await expect(assistant.getByText("2 / 2", { exact: true })).toBeVisible();

  await page.route(`**/api/chats/${id}`, async (route) => {
    if (route.request().method() === "PATCH") await route.fulfill({ status: 503,
      contentType: "application/json", body: JSON.stringify({ detail: "Settings temporarily unavailable" }) });
    else await route.continue();
  });
  const mode = page.getByRole("combobox", { name: "Generation mode" });
  await expect(mode).toHaveValue("text");
  await mode.selectOption("auto");
  await expect(page.getByText("Settings temporarily unavailable", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Dismiss error" }).click();
  // The draft keeps its selection; the settings drawer follows the saved default.
  const saved = await page.request.get(`/api/chats/${id}/metadata`);
  expect((await saved.json() as { routing_mode: string }).routing_mode).toBe("text");
  await page.getByRole("button", { name: "Turn settings" }).click();
  await expect(page.getByRole("dialog", { name: "Chat settings" })).toBeVisible();
  await expect(page.getByRole("group", { name: "Settings role" })).toHaveCount(0);
  await page.getByRole("button", { name: "Close settings" }).click();
  await page.unroute(`**/api/chats/${id}`);

  const oldest = page.locator(".message.user").filter({ hasText: "Notebook entry 0." });
  await oldest.getByRole("button", { name: "Remove this item, keep replies" }).focus();
  await oldest.getByRole("button", { name: "Remove this item, keep replies" }).press("Enter");
  await oldest.getByRole("button", { name: "Remove this item, keep replies" }).click();
  await expect(page.getByText("Message removed", { exact: true })).toHaveCount(1);
  await expect(page.locator(".messages > article.message")).toHaveCount(88);

  const lastUser = page.locator(".message.user").filter({ hasText: "One final notebook entry." });
  await lastUser.getByRole("button", { name: "Delete this turn" }).focus();
  await lastUser.getByRole("button", { name: "Delete this turn" }).press("Enter");
  await lastUser.getByRole("button", { name: "Delete turn", exact: true }).click();
  await expect(page.locator(".messages > article.message")).toHaveCount(86);
  await expect(lastUser).toHaveCount(0);
}

for (const width of [1280, 390]) {
  test(`keeps bounded transcript pages, scroll position and accepted messages at width ${width}`, async ({ page, request }) => {
    test.setTimeout(180_000);
    await page.setViewportSize({ width, height: 844 });
    const headers = await session(request);
    const created = await request.post("/api/chats", { headers,
      data: { title: `Transcript pages ${width}`, routing_mode: "text" } });
    expect(created.status()).toBe(201);
    const { id } = await created.json() as { id: string };
    try {
      for (let index = 0; index < 43; index++) {
        const accepted = await request.post(`/api/chats/${id}/turns`, { headers,
          data: { text: `Notebook entry ${index}.`, mode: "text" } });
        expect(accepted.status()).toBe(202);
        await expect.poll(async () => {
          const response = await request.get(`/api/chats/${id}/context`);
          expect(response.ok()).toBeTruthy();
          return (await response.json() as { has_pending_response: boolean }).has_pending_response;
        }, { intervals: [20, 50, 100] }).toBe(false);
      }
      await page.addInitScript((chatId) => {
        localStorage.setItem("local-lm-chat", chatId);
        sessionStorage.setItem("lm-atelier-setup-dismissed", "1");
      }, id);
      await page.emulateMedia({ reducedMotion: "reduce" });
      const reads: URL[] = [];
      const errors: string[] = [];
      page.on("pageerror", (error) => errors.push(error.message));
      page.on("request", (request) => {
        const url = new URL(request.url());
        if (request.method() === "GET" && url.pathname.startsWith(`/api/chats/${id}`)) reads.push(url);
      });
      await page.goto("/");
      const messages = page.locator(".messages > article.message");
      await expect(messages).toHaveCount(40);
      await expect(page.locator(".message.user").filter({ hasText: "Notebook entry 0." })).toHaveCount(0);
      const viewport = page.locator(".messages");
      await viewport.evaluate((element) => { element.scrollTop = 0; });
      const anchor = page.locator(".message.user").filter({ hasText: "Notebook entry 23." });
      await expect(anchor).toBeVisible();
      const anchorTop = (await anchor.boundingBox())!.y;
      let refuseOlder = true;
      await page.route(`**/api/chats/${id}/messages?**`, async (route) => {
        if (refuseOlder && new URL(route.request().url()).searchParams.has("before")) {
          await route.fulfill({ status: 503, contentType: "application/json",
            body: JSON.stringify({ detail: "Older messages temporarily unavailable" }) });
        } else await route.continue();
      });
      const older = page.getByRole("button", { name: "Load older messages", exact: true });
      await older.focus();
      await older.press("Enter");
      await expect(page.getByText("Older messages temporarily unavailable", { exact: true })).toBeVisible();
      await expect(messages).toHaveCount(40);
      await expect(older).toBeFocused();
      refuseOlder = false;
      await older.press("Enter");
      await expect(messages).toHaveCount(80);
      await expect.poll(async () => Math.abs((await anchor.boundingBox())!.y - anchorTop)).toBeLessThan(3);
      await expect(older).toBeFocused();

      await viewport.evaluate((element) => { element.scrollTop = 0; });
      await older.press("Enter");
      await expect(messages).toHaveCount(86);
      await expect(page.getByRole("button", { name: "All messages loaded" })).toBeFocused();
      await expect(page.locator(".message.user").filter({ hasText: "Notebook entry 0." })).toHaveCount(1);

      const composer = page.getByRole("textbox", { name: "Message", exact: true });
      await composer.fill("One final notebook entry.");
      await composer.press("Enter");
      await expect(messages).toHaveCount(88);
      await expect(page.locator(".message.assistant").last()).toContainText("One final notebook entry.");
      await expect(page.locator(".message.user").filter({ hasText: "Notebook entry 0." })).toHaveCount(1);
      await expect.poll(() => reads.filter((url) => url.pathname.endsWith("/messages") && url.searchParams.has("before")).length)
        .toBeGreaterThan(3);
      expect(reads.some((url) => url.pathname === `/api/chats/${id}`)).toBe(false);
      const windows = reads.filter((url) => url.pathname.endsWith("/messages"));
      expect(windows.length).toBeGreaterThan(3);
      expect(windows.every((url) => Number(url.searchParams.get("limit")) > 0 && Number(url.searchParams.get("limit")) <= 40)).toBe(true);
      await exerciseTranscriptActions(page, id);
      expect(reads.some((url) => url.pathname === `/api/chats/${id}`)).toBe(false);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
      expect(errors).toEqual([]);
    } finally {
      await page.goto("about:blank");
      expect.soft((await request.delete(`/api/chats/${id}`, { headers })).ok()).toBeTruthy();
    }
  });
}
