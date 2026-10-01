import { expect, test } from "@playwright/test";
import type { WorkflowFamily, WorkflowSelection } from "../apps/web/src/types";

const stamp = "2026-09-01T00:00:00Z";
const families: WorkflowFamily[] = Array.from({ length: 52 }, (_, index) => ({
  id: `choice-${index}`, name: `Choice ${String(index).padStart(2, "0")}`,
  description: "Neutral workflow choice", use_case: "", tags: [],
  enabled: true, archived: false, compatibility: false, created_at: stamp, updated_at: stamp,
  preferences: [{ selector_capability: "image", enabled: true, is_default: false, sort_order: index }],
  variant_count: 1, ready_variant_count: 1, best_readiness: "ready",
  variants: [{ id: `workflow-${index}`, name: "Example", variant_key: "image", operation: "text_to_image",
    current_revision_id: `revision-${index}`, current_revision_version: 1, engine: "mock",
    capabilities: ["image"], trusted: true, readiness: "ready", readiness_reason: null }],
}));

for (const width of [1280, 375]) {
  test(`chat workflow choices keep an off-page selection at ${width}px`, async ({ page, request }, testInfo) => {
    await page.setViewportSize({ width, height: 900 });
    const session = await request.post("/api/session");
    expect(session.ok()).toBeTruthy();
    const { csrf_token } = await session.json() as { csrf_token: string };
    const created = await request.post("/api/chats", {
      headers: { "x-local-lm-csrf": csrf_token }, data: { title: `Workflow choices ${width}` },
    });
    expect(created.status()).toBe(201);
    const { id } = await created.json() as { id: string };
    await page.addInitScript(chatId => localStorage.setItem("local-lm-chat", chatId), id);
    let current: WorkflowSelection = {
      selector_capability: "image", mode: "family", workflow_family_id: "choice-51", workflow_revision_id: null,
      legacy_profile_id: null,
    };
    const writes: unknown[] = [];
    const reads: URLSearchParams[] = [];
    let releasePage: (() => void) | undefined;
    let failSearch = true;
    await page.route(`**/api/chats/${id}/workflow-selections`, route => route.fulfill({ json: [current] }));
    await page.route(`**/api/chats/${id}/workflow-selections/image`, route => {
      const choice = route.request().postDataJSON() as Pick<WorkflowSelection, "mode" | "workflow_family_id">;
      writes.push(choice);
      current = { ...current, ...choice };
      return route.fulfill({ json: current });
    });
    await page.route(/\/api\/workflow-families\?/, async route => {
      const query = new URL(route.request().url()).searchParams;
      reads.push(query);
      if (query.get("selector_capability") !== "image") return route.fulfill({ json: [] });
      const search = query.get("search")?.toLowerCase() ?? "";
      if (search === "choice 10" && failSearch) {
        return route.fulfill({ status: 503, json: { detail: "Workflow search unavailable" } });
      }
      if (query.get("offset") === "50") await new Promise<void>(resolve => { releasePage = resolve; });
      const ids = query.getAll("family_ids");
      const rows = families.filter(family => (!ids.length || ids.includes(family.id))
        && (!search || family.name.toLowerCase().includes(search)));
      const offset = Number(query.get("offset") ?? 0);
      return route.fulfill({ json: rows.slice(offset, offset + Number(query.get("limit") ?? rows.length)) });
    });
    await page.goto("/");
    const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
    await expect(setup).toBeVisible();
    await setup.getByRole("button", { name: "Not now" }).click();
    const select = page.getByRole("combobox", { name: "Image workflow", exact: true });
    await expect(select).toHaveValue("choice-51");
    await expect(select.getByRole("option", { name: "Choice 51", exact: true })).toBeAttached();
    await expect(select.getByRole("option", { name: "Choice 50", exact: true })).toHaveCount(0);
    expect(reads.some(query => query.get("family_ids") === "choice-51" && query.get("limit") === "1")).toBe(true);
    const load = page.getByRole("button", { name: "Load more image workflows", exact: true });
    await load.click();
    const loading = page.getByRole("button", { name: "Loading more image workflows…", exact: true });
    await expect(loading).toHaveAttribute("aria-disabled", "true");
    await expect(loading).toBeFocused();
    await expect(select).toHaveValue("choice-51");
    await expect.poll(() => Boolean(releasePage)).toBe(true);
    releasePage!();
    await expect(select.getByRole("option", { name: "Choice 50", exact: true })).toBeAttached();
    const search = page.getByRole("searchbox", { name: "Search image workflows", exact: true });
    await search.fill("Choice 10");
    const retry = page.getByRole("button", { name: "Retry image workflows", exact: true });
    await expect(retry).toBeVisible();
    await expect(select).toHaveValue("choice-51");
    await expect(search).toBeFocused();
    expect(writes).toEqual([]);
    failSearch = false;
    await retry.click();
    await expect(select.getByRole("option", { name: "Choice 10", exact: true })).toBeAttached();
    await expect(select.getByRole("option", { name: "Choice 00", exact: true })).toHaveCount(0);
    await expect(select).toHaveValue("choice-51");
    await select.selectOption("choice-10");
    await expect(select).toHaveValue("choice-10");
    expect(writes).toEqual([{ mode: "family", workflow_family_id: "choice-10" }]);
    expect(reads.every(query => Number(query.get("limit")) > 0 && Number(query.get("limit")) <= 50
      && Number(query.get("variant_limit")) > 0 && Number(query.get("variant_limit")) <= 2)).toBe(true);
    await select.scrollIntoViewIfNeeded();
    const bounds = await select.boundingBox();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(width);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    await page.screenshot({ path: testInfo.outputPath("workflow-selector-pages.png"), fullPage: true });
  });
}
