import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import type { Workflow, WorkflowFamily } from "../apps/web/src/types";

const stamp = "2026-09-01T00:00:00Z";
const families: WorkflowFamily[] = Array.from({ length: 12 }, (_, index) => ({
  id: `family-${index}`, name: `Collection ${String(index).padStart(2, "0")}`,
  description: "Neutral workflow collection", use_case: "", tags: index === 11 ? ["Distant example"] : [],
  enabled: true, archived: false, compatibility: false, created_at: stamp, updated_at: stamp,
  preferences: [], variants: Array.from({ length: 7 }, (_, variant) => ({
    id: `workflow-${index}-${variant}`, name: `Example ${index}/${variant}`, variant_key: `variant-${variant}`,
    operation: index === 11 ? "text_to_video" : "text_to_image",
    current_revision_id: `revision-${index}-${variant}`, current_revision_version: 1,
    engine: "mock", capabilities: [index === 11 ? "video" : "image"],
    trusted: true, readiness: "ready", readiness_reason: null,
  })),
}));

function familyPage(query: URLSearchParams) {
  const search = query.get("search")?.toLowerCase() ?? "";
  const ids = query.getAll("family_ids");
  const workflows = query.getAll("workflow_ids");
  const operation = query.get("operation");
  const matching = families.filter(family => (!ids.length || ids.includes(family.id))
    && (!workflows.length || family.variants.some(variant => workflows.includes(variant.id)))
    && (!search || [family.name, ...family.tags].some(value => value.toLowerCase().includes(search))))
    .map(family => {
      const variants = family.variants.filter(variant => (!operation || variant.operation === operation)
        && (!workflows.length || workflows.includes(variant.id)));
      const offset = Number(query.get("variant_offset") ?? 0);
      const limit = Number(query.get("variant_limit") ?? variants.length);
      return { ...family, variant_count: variants.length, ready_variant_count: variants.length,
        best_readiness: "ready", variants: variants.slice(offset, offset + limit) };
    }).filter(family => family.variant_count > 0);
  const offset = Number(query.get("offset") ?? 0);
  return matching.slice(offset, offset + Number(query.get("limit") ?? matching.length));
}

function workflow(id: string): Workflow {
  const family = families.find(row => row.variants.some(variant => variant.id === id))!;
  const variant = family.variants.find(row => row.id === id)!;
  return {
    id, family_id: family.id, name: variant.name, description: "Neutral example", operation: variant.operation,
    current_revision_id: variant.current_revision_id,
    revisions: [{ id: variant.current_revision_id!, workflow_id: id, version: 1, engine: "mock",
      engine_version: null, trusted: true, created_at: stamp,
      api_graph_json: {}, ui_graph_json: {}, input_schema_json: {}, dependencies_json: {} }],
  };
}

async function openLibrary(page: Page, width: number) {
  await page.setViewportSize({ width, height: 900 });
  await page.goto("/");
  const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setup).toBeVisible();
  await setup.getByRole("button", { name: "Not now" }).click();
  if (width < 700) await page.getByRole("button", { name: "Toggle navigation" }).click();
  await page.getByRole("button", { name: "Workflows", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Collection 00", exact: true })).toBeVisible();
}

for (const width of [1280, 375]) {
  test(`workflow pages keep selections and global filters at ${width}px`, async ({ page }, testInfo) => {
    const reads: URLSearchParams[] = [];
    const details: string[] = [];
    let failNextPage = true;
    let releasePage: (() => void) | undefined;
    let holdPage = false;
    await page.route(/\/api\/workflow-families\?/, async route => {
      const query = new URL(route.request().url()).searchParams;
      reads.push(query);
      if (query.get("offset") === "10") {
        if (failNextPage) return route.fulfill({ status: 503, json: { detail: "Family page unavailable" } });
        if (holdPage) {
          holdPage = false;
          await new Promise<void>(resolve => { releasePage = resolve; });
        }
      }
      await route.fulfill({ json: familyPage(query) });
    });
    await page.route("**/api/workflow-family-operations", route => route.fulfill({
      json: ["text_to_image", "text_to_video"],
    }));
    await page.route("**/api/workflow-summaries?*", route => route.fulfill({ json: [] }));
    await page.route(/\/api\/workflows\/workflow-\d+-\d+$/, route => {
      const id = new URL(route.request().url()).pathname.split("/").at(-1)!;
      details.push(id);
      return route.fulfill({ json: workflow(id) });
    });
    await openLibrary(page, width);
    const first = page.getByRole("region", { name: "Collection 00", exact: true });
    await expect(first.getByRole("button", { name: /Example 0\/6/ })).toHaveCount(0);
    await expect(page.getByRole("heading", { name: "Collection 11", exact: true })).toHaveCount(0);
    await expect(page.getByRole("option", { name: "Text to video", exact: true })).toBeAttached();
    expect(details).toEqual([]);

    await first.getByRole("button", { name: "Load more Collection 00 variants" }).click();
    const selected = first.getByRole("button", { name: /Example 0\/6/ });
    await selected.click();
    const heading = page.getByRole("heading", { name: "Example 0/6", exact: true });
    await expect(heading).toBeVisible();
    await expect(selected).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("button", { name: "Show operation variants" })).toBeVisible();
    expect(details).toEqual(["workflow-0-6"]);
    expect(reads.some(query => query.get("workflow_ids") === "workflow-0-6"
      && query.get("limit") === "1" && query.get("variant_limit") === "1")).toBe(true);

    await page.getByRole("button", { name: "Load more workflow families", exact: true }).click();
    const retry = page.getByRole("button", { name: "Retry workflow families" });
    await expect(retry).toBeVisible();
    await expect(heading).toBeVisible();
    await expect(selected).toHaveAttribute("aria-pressed", "true");
    failNextPage = false;
    await retry.click();
    await expect(page.getByRole("heading", { name: "Collection 11", exact: true })).toBeVisible();

    const search = page.getByRole("searchbox", { name: "Search workflow families" });
    await search.fill("Distant example");
    await expect(first).toHaveCount(0);
    await expect(page.getByRole("heading", { name: "Collection 11", exact: true })).toBeVisible();
    await expect(heading).toBeVisible();
    await expect(search).toBeFocused();
    await search.fill("");
    await expect(first).toBeVisible();
    holdPage = true;
    const load = page.getByRole("button", { name: "Load more workflow families", exact: true });
    // A new filter starts a fresh page sequence instead of reusing every previous page.
    await page.getByRole("combobox", { name: "Filter by readiness" }).selectOption("ready");
    await load.click();
    const loading = page.getByRole("button", { name: "Loading more workflow families…", exact: true });
    await expect(loading).toHaveAttribute("aria-disabled", "true");
    await expect(loading).toBeFocused();
    await expect.poll(() => Boolean(releasePage)).toBe(true);
    releasePage!();
    await expect(page.getByRole("heading", { name: "Collection 11", exact: true })).toBeVisible();
    await page.getByRole("combobox", { name: "Filter by operation" }).selectOption("text_to_video");
    await expect(first).toHaveCount(0);
    await expect(page.getByRole("heading", { name: "Collection 11", exact: true })).toBeVisible();
    await expect(heading).toBeVisible();
    expect(reads.every(query => {
      const selector = query.get("order") === "preference"
        && query.has("selector_capability") && query.get("enabled_only") === "true"
        && query.get("variant_limit") === "1";
      return Number(query.get("limit")) > 0
        && Number(query.get("limit")) <= (selector ? 50 : 10)
        && Number(query.get("variant_limit")) > 0 && Number(query.get("variant_limit")) <= 5;
    })).toBe(true);
    expect(reads.some(query => query.get("variant_offset") === "5")).toBe(true);
    expect(reads.some(query => query.get("search") === "Distant example")).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    await search.scrollIntoViewIfNeeded();
    await page.screenshot({ path: testInfo.outputPath("workflow-library-pages.png"), fullPage: true });
  });
}
