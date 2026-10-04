import { expect, test, type Page } from "@playwright/test";
import { permanentlyDeleteChat } from "./recovery-cleanup";
import { lockedHeading, openWorkspaceForOthers } from "./workspace-lock-fixtures";
import type { ChatSummary } from "../apps/web/src/types";

const TITLE = "Harbor survey notes";
const PROMPT = "Describe a blue ceramic bowl in one sentence.";

test.afterAll(async ({ request }) => {
  await openWorkspaceForOthers(request);
});

/** What a person or a screen reader can read from the page, with its title and the whole document. */
async function readable(page: Page) {
  const body = page.locator("body");
  return [await page.title(), await body.innerText(), await body.ariaSnapshot(), await page.content()].join("\n");
}

async function expectNothingOf(page: Page, words: string[]) {
  await expect(lockedHeading(page)).toBeVisible({ timeout: 10_000 });
  const text = await readable(page);
  for (const word of words) expect(text).not.toContain(word);
}

test("a locked window shows none of the work, however it is reached", async ({ browser, request }) => {
  test.setTimeout(90_000);
  const session = await request.post("/api/session");
  const headers = { "x-local-lm-csrf": (await session.json() as { csrf_token: string }).csrf_token };
  const imported = await request.post("/api/models/import", { headers, data: {
    name: "Lock contents fixture", role: "chat", engine: "mock", local_path: process.env.LM_ATELIER_E2E_MODEL_PATH,
  } });
  expect(imported.status()).toBe(201);
  const created = await request.post("/api/chats", { headers, data: { title: TITLE, routing_mode: "text" } });
  expect(created.status()).toBe(201);
  const chatId = (await created.json() as { id: string }).id;
  const context = await browser.newContext();
  try {
    const accepted = await request.post(`/api/chats/${chatId}/turns`, { headers, data: { text: PROMPT, mode: "text" } });
    expect(accepted.status()).toBe(202);
    await expect.poll(async () => {
      const response = await request.get("/api/chats/summaries?limit=50");
      expect(response.status()).toBe(200);
      return (await response.json() as ChatSummary[]).find((chat) => chat.id === chatId)?.activity.last_output?.id;
    }).toBeTruthy();

    await context.addInitScript((id) => {
      localStorage.setItem("local-lm-chat", id);
      sessionStorage.setItem("lm-atelier-setup-dismissed", "1");
    }, chatId);
    const work = await context.newPage();
    let locking = false;
    const lateFrames: string[] = [];
    work.on("websocket", (socket) => {
      socket.on("framereceived", ({ payload }) => {
        if (locking) lateFrames.push(String(payload));
      });
    });
    await work.goto("/");
    const transcript = work.getByRole("region", { name: TITLE, exact: true });
    await expect(transcript).toBeVisible();
    const answer = work.locator(".messages > .message.assistant .message-content p").first();
    await expect(answer).not.toBeEmpty();
    const reply = (await answer.innerText()).trim();
    expect(reply.length).toBeGreaterThan(8);
    const words = [TITLE, PROMPT, reply];

    const settings = await context.newPage();
    await settings.goto("/?view=settings&settings=privacy");
    const on = settings.getByRole("button", { name: "On", exact: true });
    await on.click();
    await expect(on).toHaveAttribute("aria-pressed", "true");
    locking = true;
    await settings.getByRole("button", { name: "Lock now", exact: true }).click();
    await expectNothingOf(settings, words);

    // The window that was showing the conversation learns of the lock by itself.
    await expectNothingOf(work, words);
    const refused = await work.evaluate(async () => {
      const response = await fetch("/api/chats/summaries?limit=50");
      return { status: response.status, body: await response.text() };
    });
    expect(refused.status).toBe(423);
    for (const word of words) expect(refused.body).not.toContain(word);

    await work.reload();
    await expectNothingOf(work, words);
    await work.goto("/?view=media");
    await expectNothingOf(work, words);
    await work.goBack();
    await expectNothingOf(work, words);
    await work.goForward();
    await expectNothingOf(work, words);
    const fresh = await context.newPage();
    await fresh.goto("/?view=chat");
    await expectNothingOf(fresh, words);
    for (const frame of lateFrames) {
      for (const word of words) expect(frame).not.toContain(word);
    }

    // Unlocking brings the conversation back as it was.
    await fresh.getByRole("button", { name: "Unlock", exact: true }).click();
    await expect(lockedHeading(fresh)).toHaveCount(0);
    await expect(fresh.getByRole("region", { name: TITLE, exact: true })).toBeVisible();
    await expect(fresh.locator(".messages > .message.assistant .message-content p").first()).toHaveText(reply);
  } finally {
    await context.close();
    await openWorkspaceForOthers(request);
    await permanentlyDeleteChat(request, chatId, headers);
  }
});
