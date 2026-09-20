import { expect, test } from "@playwright/test";
import type { ChatSummary } from "../apps/web/src/types";

const seenKey = "lm-atelier:chat-activity-seen:v1";

test.beforeAll(async ({ request }) => {
  const session = await request.post("/api/session");
  const { csrf_token: token } = await session.json() as { csrf_token: string };
  const modelPath = process.env.LM_ATELIER_E2E_MODEL_PATH;
  expect(modelPath).toBeTruthy();
  const imported = await request.post("/api/models/import", {
    headers: { "x-local-lm-csrf": token },
    data: { name: "Color study fixture", role: "chat", engine: "mock", local_path: modelPath },
  });
  expect(imported.status()).toBe(201);
});

for (const width of [1280, 390]) {
  test(`acknowledges completed output only after visible rendering at width ${width}`, async ({ page, request }) => {
    await page.setViewportSize({ width, height: 844 });
    const session = await request.post("/api/session");
    const { csrf_token: token } = await session.json() as { csrf_token: string };
    const headers = { "x-local-lm-csrf": token };
    const chatIds: string[] = [];
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    try {
      for (const title of ["Finished color study", "Reading notebook"]) {
        const created = await request.post("/api/chats", { headers, data: { title, routing_mode: "text" } });
        expect(created.status()).toBe(201);
        chatIds.push((await created.json() as { id: string }).id);
      }
      const [completedId, readingId] = chatIds;
      const accepted = await request.post(`/api/chats/${completedId}/turns`, {
        headers, data: { text: "Describe a blue ceramic bowl in one sentence.", mode: "text" },
      });
      expect(accepted.status()).toBe(202);
      let completed: ChatSummary | undefined;
      await expect.poll(async () => {
        const response = await request.get("/api/chats/summaries?limit=50");
        expect(response.status()).toBe(200);
        completed = (await response.json() as ChatSummary[]).find((chat) => chat.id === completedId);
        return completed?.activity.last_output?.id;
      }).toBeTruthy();
      const activity = completed!.activity.last_output!;
      await page.addInitScript((id) => {
        localStorage.setItem("local-lm-chat", id);
        sessionStorage.setItem("lm-atelier-setup-dismissed", "1");
      }, readingId);
      await page.goto("/");
      await expect(page.getByRole("heading", { name: "Reading notebook", exact: true })).toBeVisible();
      if (width < 600) await page.getByRole("button", { name: "Toggle navigation" }).click();
      const row = page.locator(".sidebar-chat-row").filter({ hasText: completed!.title });
      await expect(row.getByRole("img", { name: "Unread output" })).toBeVisible();
      const transparent = await page.addStyleTag({ content: ".messages [data-chat-activity] { opacity: 0; }" });
      await row.locator(".chat-main").focus();
      await page.keyboard.press("Enter");
      const rendered = page.locator("[data-chat-activity]");
      await expect(rendered).toHaveCount(1);
      await expect(rendered).toHaveCSS("opacity", "0");
      await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
      expect(await page.evaluate((key) => localStorage.getItem(key), seenKey)).toBeNull();
      await expect(row.getByRole("img", { name: "Unread output", includeHidden: true })).toHaveCount(1);
      await transparent.evaluate((element) => element.parentNode?.removeChild(element));
      await expect(rendered).toHaveCSS("opacity", "1");
      await expect.poll(() => page.evaluate((key) => JSON.parse(localStorage.getItem(key) ?? "[]") as unknown[], seenKey))
        .toEqual([[completedId, activity.id, activity.sequence, expect.any(Number)]]);
      await expect(row.getByRole("img", { name: "Unread output", includeHidden: true })).toHaveCount(0);
      await page.reload();
      await expect(page.getByRole("heading", { name: "Reading notebook", exact: true })).toBeVisible();
      await expect(row.getByRole("img", { name: "Unread output", includeHidden: true })).toHaveCount(0);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
      expect(errors).toEqual([]);
    } finally {
      await page.goto("about:blank");
      for (const id of chatIds) expect.soft((await request.delete(`/api/chats/${id}`, { headers })).ok()).toBeTruthy();
    }
  });
}
