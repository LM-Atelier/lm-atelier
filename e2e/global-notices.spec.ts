import { expect, test, type APIRequestContext, type WebSocketRoute } from "@playwright/test";
import { permanentlyDeleteChat } from "./recovery-cleanup";

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
    data: { name: "Notice fixture", role: "chat", engine: "mock", local_path: modelPath } });
  expect(imported.status()).toBe(201);
});

const scenarios = [1280, 390].flatMap((width) =>
  [1, 1.5].flatMap((textScale) =>
    [false, true].map((disconnected) => ({ width, textScale, disconnected })),
  ),
);

for (const { width, textScale, disconnected } of scenarios) {
  test(`keeps settings controls and notices accessible at width ${width}, text scale ${textScale}, disconnected ${disconnected}`, async ({ page, request }) => {
    await page.setViewportSize({ width, height: 844 });
    const headers = await session(request);
    const created = await request.post("/api/chats", { headers,
      data: { title: `Notice controls ${width}`, routing_mode: "text" } });
    expect(created.status()).toBe(201);
    const { id } = await created.json() as { id: string };
    try {
      await page.addInitScript((chatId) => {
        localStorage.setItem("local-lm-chat", chatId);
        sessionStorage.setItem("lm-atelier-setup-dismissed", "1");
      }, id);
      let socket: WebSocketRoute | undefined;
      let offline = false;
      await page.routeWebSocket(/\/api\/events\?/, async (route) => {
        if (offline) await route.close();
        else {
          socket = route;
          route.connectToServer();
        }
      });
      await page.route(`**/api/chats/${id}`, async (route) => {
        if (route.request().method() === "PATCH") await route.fulfill({ status: 503,
          json: { detail: "Settings temporarily unavailable" } });
        else await route.continue();
      });
      await page.goto("/");
      await page.evaluate((scale) => {
        document.documentElement.style.setProperty("--text-scale", String(scale));
      }, textScale);
      await page.getByRole("combobox", { name: "Generation mode" }).selectOption("auto");
      const failure = page.getByRole("alert").filter({ hasText: "Settings temporarily unavailable" });
      await expect(failure).toBeVisible();
      if (disconnected) {
        await expect.poll(() => Boolean(socket)).toBe(true);
        offline = true;
        await socket!.close();
        const status = page.getByRole("status").filter({ hasText: "Live updates are disconnected" });
        await expect(status).toBeVisible();
        const statusBox = (await status.boundingBox())!;
        const failureBox = (await failure.boundingBox())!;
        expect(failureBox.y).toBeGreaterThanOrEqual(statusBox.y + statusBox.height);
      }
      const composerBox = (await page.locator(".composer-wrap").boundingBox())!;
      const firstNotice = disconnected
        ? page.getByRole("status").filter({ hasText: "Live updates are disconnected" })
        : failure;
      const railBox = (await firstNotice.boundingBox())!;
      expect(composerBox.y + composerBox.height).toBeLessThanOrEqual(railBox.y);
      await page.getByRole("button", { name: "Turn settings" }).click();
      const dialog = page.getByRole("dialog", { name: "Chat settings" });
      await expect(dialog).toBeVisible();
      const close = dialog.getByRole("button", { name: "Close settings" });
      const closeBox = (await close.boundingBox())!;
      const failureBox = (await failure.boundingBox())!;
      expect(failureBox.y).toBeGreaterThan(closeBox.y + closeBox.height);
      expect(failureBox.x).toBeGreaterThanOrEqual(0);
      expect(failureBox.x + failureBox.width).toBeLessThanOrEqual(width);
      if (disconnected) {
        const status = page.getByRole("status").filter({ hasText: "Live updates are disconnected" });
        await expect(status).toBeVisible();
        const statusBox = (await status.boundingBox())!;
        expect(statusBox.y).toBeGreaterThan(closeBox.y + closeBox.height);
        expect(failureBox.y).toBeGreaterThanOrEqual(statusBox.y + statusBox.height);
        expect(failureBox.y + failureBox.height).toBeLessThanOrEqual(844);
      }
      await close.click();
      await expect(dialog).toHaveCount(0);
      await expect(failure).toBeVisible();
      await expect(page.getByRole("button", { name: "Turn settings" })).toBeFocused();
      await page.getByRole("button", { name: "Dismiss error" }).click();
      await expect(failure).toHaveCount(0);
      if (disconnected) await expect(page.getByRole("status").filter({ hasText: "Live updates are disconnected" })).toBeVisible();
      else await expect.poll(async () => {
        const workspace = (await page.locator("main").boundingBox())!;
        return Math.round(workspace.y + workspace.height);
      }).toBe(844);
    } finally {
      await page.goto("about:blank");
      await permanentlyDeleteChat(request, id, headers);
    }
  });
}
