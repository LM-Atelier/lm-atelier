import { permanentlyDeleteChat } from "./recovery-cleanup";
import { expect, test } from "@playwright/test";
import type { EditedBranch, PriorTurnEditAccepted, TurnAccepted } from "../apps/web/src/types";

for (const width of [1280, 390]) {
  test(`keeps edited previews isolated and continues their history at width ${width}`, async ({ page, request }) => {
    test.setTimeout(180_000);
    await page.setViewportSize({ width, height: 844 });
    const session = await request.post("/api/session");
    const { csrf_token: csrf } = await session.json() as { csrf_token: string };
    const headers = { "x-local-lm-csrf": csrf };
    const imported = await request.post("/api/models/import", { headers, data: {
      name: "Branch fixture", role: "chat", engine: "mock", local_path: process.env.LM_ATELIER_E2E_MODEL_PATH,
    } });
    expect(imported.status()).toBe(201);
    const created = await request.post("/api/chats", { headers,
      data: { title: `Transcript branches ${width}`, routing_mode: "text" } });
    expect(created.status()).toBe(201);
    const { id } = await created.json() as { id: string };
    let release: (() => void) | undefined;
    try {
      let source: TurnAccepted | undefined;
      for (let index = 0; index < 24; index++) {
        const response = await request.post(`/api/chats/${id}/turns`, { headers,
          data: { text: `Branch notebook ${index}.`, mode: "text" } });
        expect(response.status()).toBe(202);
        source = await response.json() as TurnAccepted;
        await expect.poll(async () => {
          const context = await request.get(`/api/chats/${id}/context`);
          return (await context.json() as { has_pending_response: boolean }).has_pending_response;
        }, { intervals: [20, 50, 100] }).toBe(false);
      }
      const sourceId = source!.user_message.id;
      const originalHead = source!.assistant_message.id;
      const reads: URL[] = [];
      const errors: string[] = [];
      page.on("pageerror", (error) => errors.push(error.message));
      page.on("request", (outgoing) => {
        const url = new URL(outgoing.url());
        if (outgoing.method() === "GET" && url.pathname.startsWith(`/api/chats/${id}`)) reads.push(url);
      });
      await page.addInitScript((chatId) => {
        localStorage.setItem("local-lm-chat", chatId);
        sessionStorage.setItem("lm-atelier-setup-dismissed", "1");
      }, id);
      await page.emulateMedia({ reducedMotion: "reduce" });
      await page.route("**/api/jobs/activity?*", route => route.fulfill({ json: {
        active: [], active_count: 0, recent_issues: [{ id: "branch-fixture-issue",
          kind: "generation", status: "failed", phase: "Temporary fixture failure", error: null,
          updated_at: "2026-09-01T00:00:00Z", cancellable: false }],
      } }));
      await page.goto("/");
      await expect(page.getByRole("complementary", { name: "Jobs", exact: true })).toBeVisible();
      const active = page.locator(".messages > article.message");
      await expect(active).toHaveCount(40);
      await page.locator(".messages .message.user").last().getByRole("button", { name: "Edit message" }).click();
      const editor = page.getByRole("dialog", { name: "Queue edited version", exact: true });
      await editor.getByRole("textbox", { name: "Message", exact: true }).fill("Alternate notebook A.");
      const accepted = page.waitForResponse((response) => response.url().endsWith(`/messages/${sourceId}/edits`)
        && response.request().method() === "POST");
      await editor.getByRole("button", { name: "Queue edited version", exact: true }).click();
      const firstResponse = await accepted;
      expect(firstResponse.status()).toBe(202);
      const first = await firstResponse.json() as PriorTurnEditAccepted;
      await expect(editor).toHaveCount(0);
      const secondResponse = await request.post(`/api/messages/${sourceId}/edits`, { headers,
        data: { text: "Alternate notebook B.", idempotency_key: `browser-alternate-${width}` } });
      expect(secondResponse.status()).toBe(202);
      const second = await secondResponse.json() as PriorTurnEditAccepted;
      let branches: EditedBranch[] = [];
      await expect.poll(async () => {
        const response = await request.get(`/api/chats/${id}/edited-branches`);
        branches = (await response.json() as { items: EditedBranch[] }).items;
        return branches.length === 2 && branches.every((branch) => branch.can_continue);
      }).toBe(true);
      const metadata = await request.get(`/api/chats/${id}/metadata`);
      expect((await metadata.json() as { active_head_message_id: string }).active_head_message_id).toBe(originalHead);
      await page.reload();
      const card = (planId: string) => page.getByRole("article", {
        name: `Edited version ${branches.findIndex((branch) => branch.plan.id === planId) + 1}`, exact: true,
      });
      let held = false;
      let delivered = false;
      await page.route(`**/api/chats/${id}/messages?**`, async (route) => {
        const url = new URL(route.request().url());
        if (!held && url.searchParams.get("head_id") === first.branch_head_message_id) {
          held = true;
          const response = await route.fetch();
          await new Promise<void>((resolve) => { release = resolve; });
          // Switching previews cancels this request, but its response can still arrive late.
          await route.fulfill({ response }).catch(() => undefined);
          delivered = true;
        } else await route.continue();
      });
      await card(first.work_plan_id).getByRole("button", { name: "View edited branch" }).click();
      const preview = page.getByRole("dialog", { name: "Edited branch preview" });
      await expect(preview.getByText("This edited version is being loaded.")).toBeVisible();
      await expect.poll(() => Boolean(release)).toBe(true);
      await preview.getByRole("button", { name: "Close preview" }).click();
      await card(second.work_plan_id).getByRole("button", { name: "View edited branch" }).click();
      await expect(preview.locator("article.message")).toHaveCount(40);
      await expect(preview.locator(".message.user").last()).toHaveText(/Alternate notebook B\./);
      release!();
      await expect.poll(() => delivered).toBe(true);
      await expect(preview.locator(".message.user").last()).toHaveText(/Alternate notebook B\./);
      await expect(preview.getByText("Alternate notebook A.", { exact: true })).toHaveCount(0);
      await expect(page.locator(".messages .message.user").last()).toHaveText(/Branch notebook 23\./);

      const older = preview.getByRole("button", { name: "Load older preview messages" });
      await older.focus();
      const olderControl = await older.elementHandle();
      await older.press("Enter");
      await expect(preview.locator("article.message")).toHaveCount(48);
      await expect.poll(() => olderControl!.evaluate((element) => document.activeElement === element)).toBe(true);
      await expect(preview.getByText("Branch notebook 0.", { exact: true })).toBeVisible();
      await preview.getByRole("button", { name: "Continue from this version" }).click();
      await expect(preview).toHaveCount(0);
      await expect(active).toHaveCount(48);
      await expect(page.locator(".messages .message.user").last()).toHaveText(/Alternate notebook B\./);
      const composer = page.getByRole("textbox", { name: "Message", exact: true });
      await composer.fill("Continue the alternate notebook.");
      await composer.press("Enter");
      await expect(active).toHaveCount(50);
      await expect(page.locator(".messages .message.assistant").last()).toContainText("Continue the alternate notebook.");
      await expect(page.locator(".messages").getByText("Branch notebook 0.", { exact: true })).toBeVisible();
      expect(reads.some((url) => url.pathname === `/api/chats/${id}`)).toBe(false);
      expect(reads.filter((url) => url.pathname.endsWith("/messages")).every((url) =>
        Number(url.searchParams.get("limit")) > 0 && Number(url.searchParams.get("limit")) <= 40)).toBe(true);
      expect(errors).toEqual([]);
    } finally {
      release?.();
      await page.goto("about:blank");
      await permanentlyDeleteChat(request, id, headers);
    }
  });
}
