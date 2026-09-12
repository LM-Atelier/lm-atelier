import { expect, test } from "@playwright/test";
import path from "node:path";
import type { Chat, ChatDetail, WebSearch } from "../apps/web/src/types";

for (const width of [1280, 375]) {
  test("search approval and explicit source selection at width " + width, async ({ page, request }) => {
    await page.setViewportSize({ width, height: 900 });
    const session = await request.post("/api/session");
    expect(session.ok()).toBeTruthy();
    const { csrf_token: csrf } = await session.json() as { csrf_token: string };
    const created = await request.post("/api/chats", {
      headers: { "x-local-lm-csrf": csrf }, data: { title: "Compare copper and steel for a garden structure with a long saved conversation title" },
    });
    expect(created.status()).toBe(201);
    const chat = await created.json() as Chat;
    const stamp = "2026-09-11T00:00:00Z";
    let search: WebSearch = {
      run_id: "run-search", assistant_message_id: "answer-search", job_id: "job-search",
      revision: 1, state: "awaiting_approval", query: "Compare copper and steel",
      provider: "CRW", provider_endpoint: "https://search.example.test", dispatch_after: null,
      results: [], result_count: 0, truncated: false, error_code: null,
    };
    let activeHead = "answer-search";
    let webSettings = chat.web_settings_json;
    const permissionWrites: unknown[] = [];
    let finishPermissionSave: (() => void) | undefined;
    const detail = (): ChatDetail => ({
      ...chat, web_settings_json: webSettings, active_head_message_id: activeHead, web_searches: [search],
      messages: [
        { id: "question-search", chat_id: chat.id, parent_id: null, role: "user", status: "complete",
          parts: [{ id: "question-part", position: 0, type: "text", text: "Compare these materials.",
            artifact_id: null, metadata_json: {} }], created_at: stamp, updated_at: stamp },
        { id: "answer-search", chat_id: chat.id, parent_id: "question-search", role: "assistant", status: "pending",
          parts: [], created_at: stamp, updated_at: stamp },
      ],
    });
    const edits: unknown[] = [];
    const approvals: unknown[] = [];
    const sends: string[] = [];
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("request", (outgoing) => {
      if (outgoing.method() === "POST" && outgoing.url().endsWith("/turns")) sends.push(outgoing.url());
    });
    await page.addInitScript((id) => localStorage.setItem("local-lm-chat", id), chat.id);
    await page.route("**/api/web-search/configuration", (route) => route.fulfill({ json: {
      installation_enabled: true, configured: true, provider: "CRW",
      provider_endpoint: "https://search.example.test", error_code: null,
    } }));
    await page.route("**/api/chats/" + chat.id, async (route) => {
      if (route.request().method() === "PATCH") {
        const body = route.request().postDataJSON() as { web_settings_json: typeof webSettings };
        permissionWrites.push(body);
        await new Promise<void>((resolve) => { finishPermissionSave = resolve; });
        webSettings = body.web_settings_json;
      }
      await route.fulfill({ json: detail() });
    });
    await page.route("**/api/jobs/job-search/search", async (route) => {
      const body = route.request().postDataJSON() as { revision: number; query: string };
      edits.push(body);
      search = { ...search, query: body.query, revision: search.revision! + 1 };
      await route.fulfill({ json: search });
    });
    await page.route("**/api/jobs/job-search/search/decision", async (route) => {
      approvals.push(route.request().postDataJSON());
      search = { ...search, state: "complete", revision: null, job_id: null, result_count: 1,
        results: [{ title: "Material reference", url: "https://materials.example.test/reference", snippet: "A neutral comparison." }] };
      await route.fulfill({ json: search });
    });
    await page.goto("/");
    const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
    await expect(setup).toBeVisible();
    await setup.getByRole("button", { name: "Not now" }).click();
    const query = page.getByLabel("Exact query");
    await expect(query).toHaveValue("Compare copper and steel");
    await expect(page.getByText("Provider: CRW at https://search.example.test", { exact: true })).toBeVisible();
    await expect(page.getByText("Web access", { exact: true })).toBeVisible();
    if (process.env.LM_ATELIER_E2E_SCREENSHOT_DIR) {
      await page.screenshot({ path: path.join(process.env.LM_ATELIER_E2E_SCREENSHOT_DIR, "search-consent-" + width + ".png"), fullPage: true });
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    expect(await page.locator(".chat-web-access").evaluate((element) => element.getBoundingClientRect().height)).toBeLessThan(80);
    const mainFits = () => page.locator("#main-content").evaluate(
      (element) => element.scrollWidth <= element.clientWidth,
    );
    expect(await mainFits()).toBe(true);
    const permissions = page.locator(".chat-web-access");
    await permissions.locator("summary").click();
    await expect(permissions.getByRole("checkbox", { name: "Read links I include in messages" })).toBeVisible();
    const allowSearch = permissions.getByRole("checkbox", { name: "Allow web searches", exact: true });
    await allowSearch.click();
    await expect.poll(() => permissionWrites.length).toBe(1);
    await expect(allowSearch).toBeFocused();
    await expect(permissions.locator("fieldset")).toHaveAttribute("aria-disabled", "true");
    expect(await allowSearch.evaluate((element) => element.matches(":disabled"))).toBe(false);
    await allowSearch.press("Space");
    expect(permissionWrites).toHaveLength(1);
    finishPermissionSave!();
    await expect(allowSearch).toBeChecked();
    await expect(allowSearch).toBeEnabled();
    await expect(allowSearch).toBeFocused();
    expect(await mainFits()).toBe(true);
    if (process.env.LM_ATELIER_E2E_SCREENSHOT_DIR) {
      await page.screenshot({ path: path.join(process.env.LM_ATELIER_E2E_SCREENSHOT_DIR, "search-permissions-" + width + ".png"), fullPage: true });
    }
    await permissions.locator("summary").click();

    activeHead = "question-search";
    await page.reload();
    const otherBranch = page.getByRole("region", { name: "Pending searches in other branches" });
    await expect(otherBranch).toBeVisible();
    await expect(otherBranch.getByLabel("Exact query")).toHaveValue(search.query);
    if (process.env.LM_ATELIER_E2E_SCREENSHOT_DIR) {
      await page.screenshot({ path: path.join(process.env.LM_ATELIER_E2E_SCREENSHOT_DIR, "search-other-branch-" + width + ".png"), fullPage: true });
    }
    for (const invalid of ["Copper\nsteel", "Copper\tsteel", "Copper\u2028steel", "\u200b".repeat(2001)]) {
      await query.fill(invalid);
      await expect(query).toHaveAttribute("aria-invalid", "true");
      await expect(otherBranch.getByRole("alert")).toContainText("one line");
      await otherBranch.getByRole("button", { name: "Save query", exact: true }).click({ force: true });
      await otherBranch.getByRole("button", { name: "Search", exact: true }).click({ force: true });
      expect(edits).toEqual([]);
      expect(approvals).toEqual([]);
    }
    const unicodeQuery = "\u0645\u06cc\u200c\u0631\u0648\u0645 \u2764\ufe0f";
    await query.fill(unicodeQuery);
    await expect(query).toHaveAttribute("aria-invalid", "false");
    const formatting = otherBranch.getByRole("region", { name: "Query formatting" });
    await expect(formatting).toContainText("U+200C");
    await expect(formatting).toContainText("U+FE0F");
    await expect(query).toHaveAccessibleDescription(/Character 3: non-joiner U\+200C/);
    const original = formatting.locator(".chat-search-query-text");
    await expect(original).toHaveText(unicodeQuery);
    expect(await original.evaluate((element) => {
      const text = element.firstChild!;
      const head = document.createRange();
      head.setStart(text, 0); head.setEnd(text, 2);
      const tail = document.createRange();
      tail.setStart(text, 3); tail.setEnd(text, 6);
      return tail.getBoundingClientRect().left < head.getBoundingClientRect().left;
    })).toBe(true);
    const marker = formatting.locator(".chat-search-character").first();
    await marker.scrollIntoViewIfNeeded();
    await expect(marker).toBeInViewport();
    expect(await marker.evaluate((element) => {
      const style = getComputedStyle(element);
      const box = element.getBoundingClientRect();
      return style.display !== "none" && style.visibility === "visible"
        && Number(style.opacity) > 0 && box.width > 0 && box.height > 0;
    })).toBe(true);
    expect(await mainFits()).toBe(true);
    if (process.env.LM_ATELIER_E2E_SCREENSHOT_DIR) {
      await page.screenshot({ path: path.join(process.env.LM_ATELIER_E2E_SCREENSHOT_DIR, "search-unicode-" + width + ".png"), fullPage: true });
    }
    await otherBranch.getByRole("button", { name: "Save query", exact: true }).click();
    await expect.poll(() => edits).toEqual([{ revision: 1, query: unicodeQuery }]);
    expect(approvals).toEqual([]);
    edits.splice(0);
    activeHead = "answer-search";
    await page.reload();
    await expect(otherBranch).toHaveCount(0);
    await query.fill("  Compare brass and steel  ");
    const approve = page.getByRole("button", { name: "Search", exact: true });
    await expect(approve).toHaveAttribute("aria-disabled", "true");
    await approve.click({ force: true });
    expect(approvals).toEqual([]);
    await query.focus();
    await page.keyboard.press("Tab");
    await expect(page.getByRole("button", { name: "Save query", exact: true })).toBeFocused();
    await page.keyboard.press("Enter");
    await expect.poll(() => edits).toEqual([{ revision: 2, query: "  Compare brass and steel  " }]);
    await expect(approve).toHaveAttribute("aria-disabled", "false");
    await page.keyboard.press("Tab");
    await expect(approve).toBeFocused();
    await page.keyboard.press("Enter");
    await expect.poll(() => approvals).toEqual([{ revision: 3, action: "approve" }]);
    await expect(page.getByRole("link", { name: "Material reference" })).toBeVisible();
    const message = page.getByRole("textbox", { name: "Message", exact: true });
    await message.fill("Explain the comparison.");
    await page.getByRole("button", { name: "Add source to message" }).click();
    await expect(message).toHaveValue("Explain the comparison.\n\nhttps://materials.example.test/reference");
    expect(sends).toEqual([]);
    expect(errors).toEqual([]);
  });
}
