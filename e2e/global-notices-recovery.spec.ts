import { expect, test } from "@playwright/test";
import { permanentlyDeleteChat } from "./recovery-cleanup";

for (const width of [1280, 390]) {
  test(`keeps a standalone Undo notice clear of settings controls at width ${width}`, async ({ page, request }) => {
    await page.setViewportSize({ width, height: 844 });
    await page.route("**/api/setup/readiness", route => route.fulfill({
      json: { version: 2, state: "ready", roles: [] },
    }));
    const session = await request.post("/api/session");
    expect(session.ok()).toBeTruthy();
    const headers = { "x-local-lm-csrf": (await session.json() as { csrf_token: string }).csrf_token };
    const imported = await request.post("/api/models/import", { headers, data: {
      name: "Notice recovery fixture", role: "chat", engine: "mock",
      local_path: process.env.LM_ATELIER_E2E_MODEL_PATH,
    } });
    expect(imported.status()).toBe(201);
    const title = `Notice removable chat ${width}`;
    const keptResponse = await request.post("/api/chats", { headers,
      data: { title: `Notice remaining chat ${width}`, routing_mode: "text" } });
    expect(keptResponse.status()).toBe(201);
    const kept = await keptResponse.json() as { id: string };
    const removedResponse = await request.post("/api/chats", { headers,
      data: { title, routing_mode: "text" } });
    expect(removedResponse.status()).toBe(201);
    const removed = await removedResponse.json() as { id: string };
    try {
      await page.addInitScript((id) => {
        localStorage.setItem("local-lm-chat", id);
        sessionStorage.setItem("lm-atelier-setup-dismissed", "1");
      }, removed.id);
      await page.goto("/");
      if (width < 700) await page.getByRole("button", { name: "Toggle navigation" }).click();
      await page.getByRole("button", { name: `Manage ${title}`, exact: true }).click();
      await page.getByRole("button", { name: "Delete chat", exact: true }).click();
      const confirmation = page.getByRole("dialog", { name: "Move this chat to Recently Deleted?" });
      await confirmation.getByRole("button", { name: "Move to Recently Deleted", exact: true }).click();
      await expect(confirmation).toHaveCount(0);
      const notice = page.locator(".recovery-notice").filter({ hasText: title });
      await expect(notice.getByRole("button", { name: "Undo", exact: true })).toBeVisible();
      await expect(page.getByRole("alert")).toHaveCount(0);
      await expect(page.getByText(/Live updates are disconnected/)).toHaveCount(0);
      if (width < 700) {
        const navigation = page.getByRole("button", { name: "Toggle navigation" });
        if (await navigation.getAttribute("aria-expanded") === "true") await navigation.click();
      }
      await page.getByRole("button", { name: "Turn settings" }).click();
      const drawer = page.getByRole("dialog", { name: "Chat settings" });
      const close = drawer.getByRole("button", { name: "Close settings" });
      await expect(close).toBeVisible();
      const noticeBox = (await notice.boundingBox())!;
      const closeBox = (await close.boundingBox())!;
      expect(noticeBox.y).toBeGreaterThan(closeBox.y + closeBox.height);
      expect(noticeBox.y + noticeBox.height).toBeLessThanOrEqual(844);
      await close.click();
      await expect(drawer).toHaveCount(0);
      if (width < 700) await page.getByRole("button", { name: "Toggle navigation" }).click();
      await page.getByRole("button", { name: `Manage Notice remaining chat ${width}`, exact: true }).click();
      const manager = page.getByRole("dialog", { name: "Chat settings" });
      const save = manager.getByRole("button", { name: "Save chat", exact: true });
      await expect(save).toBeVisible();
      await save.click();
      await expect(manager).toHaveCount(0);
      await notice.getByRole("button", { name: "Undo", exact: true }).click();
      await expect(notice).toHaveCount(0);
      await expect.poll(async () => (await request.get(`/api/chats/${removed.id}/metadata`)).status()).toBe(200);
      await expect.poll(async () => {
        const workspace = (await page.locator("main").boundingBox())!;
        return Math.round(workspace.y + workspace.height);
      }).toBe(844);
    } finally {
      await page.goto("about:blank");
      await permanentlyDeleteChat(request, removed.id, headers);
      await permanentlyDeleteChat(request, kept.id, headers);
    }
  });
}
