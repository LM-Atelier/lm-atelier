import { permanentlyDeleteChat } from "./recovery-cleanup";
import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

async function session(request: APIRequestContext) {
  const response = await request.post("/api/session");
  expect(response.ok()).toBeTruthy();
  return { "x-local-lm-csrf": (await response.json() as { csrf_token: string }).csrf_token };
}

test.beforeAll(async ({ request }) => {
  const headers = await session(request);
  const imported = await request.post("/api/models/import", { headers,
    data: { name: "Other window fixture", role: "chat", engine: "mock", local_path: process.env.LM_ATELIER_E2E_MODEL_PATH } });
  expect(imported.status()).toBe(201);
});

async function settled(request: APIRequestContext, id: string) {
  await expect.poll(async () => {
    const response = await request.get(`/api/chats/${id}/context`);
    return (await response.json() as { has_pending_response: boolean }).has_pending_response;
  }, { intervals: [20, 50, 100] }).toBe(false);
}

async function open(page: Page, id: string, width: number) {
  await page.setViewportSize({ width, height: 844 });
  await page.addInitScript((chatId) => {
    localStorage.setItem("local-lm-chat", chatId);
    sessionStorage.setItem("lm-atelier-setup-dismissed", "1");
  }, id);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await expect(page.locator(".messages > article.message")).toHaveCount(40);
}

async function scrollToTop(page: Page) {
  const viewport = page.locator(".messages");
  await viewport.hover();
  for (let step = 0; step < 60; step++) await page.mouse.wheel(0, -1200);
  await expect.poll(() => viewport.evaluate((element) => element.scrollTop)).toBe(0);
}

for (const width of [1280, 390]) {
  for (const loading of [false, true]) {
    test(`keeps the conversation being read when another window sends a message ${loading ? "while older messages load" : "after older messages loaded"} at width ${width}`, async ({ browser, page, request }) => {
      test.setTimeout(180_000);
      const headers = await session(request);
      const created = await request.post("/api/chats", { headers,
        data: { title: `Other window ${width}`, routing_mode: "text" } });
      expect(created.status()).toBe(201);
      const { id } = await created.json() as { id: string };
      const other = await browser.newPage();
      try {
        for (let index = 0; index < 43; index++) {
          expect((await request.post(`/api/chats/${id}/turns`, { headers,
            data: { text: `Notebook entry ${index}.`, mode: "text" } })).status()).toBe(202);
          await settled(request, id);
        }
        const errors: string[] = [];
        page.on("pageerror", (error) => errors.push(error.message));
        await open(page, id, width);
        await open(other, id, width);
        const messages = page.locator(".messages > article.message");
        const older = page.getByRole("button", { name: "Load older messages", exact: true });
        await scrollToTop(page);
        let reading = page.locator(".message.user").filter({ hasText: "Notebook entry 23." });
        let release = () => {};
        if (loading) {
          // The older page is slow, as it can be on a busy machine.
          const held = new Promise<void>((done) => { release = done; });
          await page.route(`**/api/chats/${id}/messages?**`, async (route) => {
            if (new URL(route.request().url()).searchParams.has("before")) await held;
            await route.continue();
          });
          await older.click();
        } else {
          await older.click();
          await expect(messages).toHaveCount(80);
          await scrollToTop(page);
          reading = page.locator(".message.user").filter({ hasText: "Notebook entry 4." });
        }
        await expect(reading).toBeVisible();
        const readingTop = (await reading.boundingBox())!.y;

        const composer = other.getByRole("textbox", { name: "Message", exact: true });
        await composer.fill("A message from another window.");
        await composer.press("Enter");
        await settled(request, id);
        await expect(page.locator(".message.user").filter({ hasText: "A message from another window." })).toHaveCount(1);
        release();

        await expect(messages).toHaveCount(82);
        await expect(reading).toBeVisible();
        await expect.poll(async () => Math.abs((await reading.boundingBox())!.y - readingTop)).toBeLessThan(3);
        await expect(page.locator(".transcript-history-button")).toBeFocused();
        expect(errors).toEqual([]);
      } finally {
        await other.close();
        await page.goto("about:blank");
        await permanentlyDeleteChat(request, id, headers);
      }
    });
  }
}
