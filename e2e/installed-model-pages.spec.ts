import { expect, test } from "@playwright/test";
import type { CatalogModel, ModelInstall } from "../apps/web/src/types";

const stamp = "2026-09-30T00:00:00Z";
const models: ModelInstall[] = Array.from({ length: 56 }, (_, index) => ({
  id: `installed-page-${index}`, source_id: null,
  name: `Installed page ${String(index).padStart(2, "0")}${index === 55 ? " %_" : ""}`,
  role: "image", engine: "mock", local_path: `neutral/${index}`, size_bytes: 100,
  compatibility: "likely", manifest_json: index === 54 ? { remote_id: "neutral/off-page" }
    : index === 55 ? { remote_id: "neutral/shared", workflow_template_id: "neutral-edit" } : {},
  active: true, readiness: "ready", capability_evidence: null, created_at: stamp, updated_at: stamp,
}));
const catalog: CatalogModel[] = [
  { remote_id: "neutral/off-page", name: "Installed beyond the page", workflow_template_id: null },
  { remote_id: "neutral/shared", name: "Installed edit variant", workflow_template_id: "neutral-edit" },
  { remote_id: "neutral/shared", name: "Available create variant", workflow_template_id: "neutral-create" },
].map(item => ({ ...item, author: "Neutral", pipeline_tag: null, library_name: null,
  downloads: 0, likes: 0, trending_score: null, last_modified: null, created_at: null,
  architecture: null, parameter_count: null, license_id: null, gated: false, private: false,
  tags: [], formats: [], quantizations: [], total_size_bytes: null, compatibility: "likely",
  compatibility_reasons: [], provider: "huggingface" }));

for (const width of [1280, 375]) {
  test(`pages installed models without losing off-page catalog status at ${width}px`, async ({ page }) => {
    test.setTimeout(90_000);
    await page.setViewportSize({ width, height: 812 });
    const reads: URL[] = [];
    let holdStatus = true;
    let failStatus = true;
    let releaseStatus!: () => void;
    const statusGate = new Promise<void>(resolve => { releaseStatus = resolve; });
    let failNextPage = true;
    let holdNextPage = true;
    let releasePage: (() => void) | undefined;
    page.on("request", request => {
      const url = new URL(request.url());
      if (request.method() === "GET" && ["/api/models", "/api/profiles", "/api/models/catalog-matches"].includes(url.pathname)) reads.push(url);
    });
    await page.route(/\/api\/models(?:\?|$)/, async route => {
      const query = new URL(route.request().url()).searchParams;
      const matching = models.filter(model => model.name.toLowerCase().includes((query.get("search") ?? "").toLowerCase()));
      const offset = Number(query.get("offset") ?? 0);
      if (offset === 50) {
        if (holdNextPage) {
          holdNextPage = false;
          await new Promise<void>(resolve => { releasePage = resolve; });
        }
        if (failNextPage) return route.fulfill({ status: 503, json: { detail: "Installed page temporarily unavailable" } });
      }
      await route.fulfill({ json: matching.slice(offset, offset + Number(query.get("limit") ?? matching.length)) });
    });
    await page.route(/\/api\/profiles(?:\?|$)/, route => route.fulfill({ json: [] }));
    await page.route("**/api/models/catalog-matches?*", async route => {
      if (holdStatus) await statusGate;
      if (failStatus) return route.fulfill({ status: 503, json: { detail: "Installation status temporarily unavailable" } });
      const query = new URL(route.request().url()).searchParams;
      const remoteIds = new Set(models.map(model => model.manifest_json.remote_id));
      const templateIds = new Set(models.map(model => model.manifest_json.workflow_template_id));
      await route.fulfill({ json: {
        remote_ids: query.getAll("remote_id").filter(id => remoteIds.has(id)),
        workflow_template_ids: query.getAll("workflow_template_id").filter(id => templateIds.has(id)),
      } });
    });
    await page.route(/\/api\/catalog\?/, route => route.fulfill({ json: { items: catalog, next_cursor: null, stale: false } }));
    await page.route("**/api/models/updates", route => route.fulfill({ json: [] }));
    await page.route("**/api/model-assets", route => route.fulfill({ json: [] }));
    await page.route("**/api/models/storage", route => route.fulfill({ json: {
      installed_count: models.length, installed_bytes: 5600, partial_download_count: 0,
      partial_download_bytes: 0, catalog_cache_bytes: 0,
    } }));
    await page.addInitScript(() => sessionStorage.setItem("lm-atelier-setup-dismissed", "1"));
    try {
      await page.goto("/?view=models");
      await page.getByRole("combobox", { name: "Model role", exact: true }).selectOption("image");
      const rows = page.locator(".profile-table.model-installs > div");
      await expect(rows).toHaveCount(50);
      await expect(page.getByText("Installed page 55 %_", { exact: true })).toHaveCount(0);
      const cards = page.locator("article.model-card");
      await expect(cards).toHaveCount(3);
      await expect(cards.getByRole("button", { name: "Checking installation…", exact: true })).toHaveCount(3);
      expect(await cards.getByRole("button", { name: "Install", exact: true }).count()).toBe(0);
      holdStatus = false;
      releaseStatus();
      await expect(cards.getByRole("button", { name: "Status unavailable", exact: true })).toHaveCount(3);
      failStatus = false;
      await page.getByRole("button", { name: "Retry installation status" }).click();
      for (const name of ["Installed beyond the page", "Installed edit variant"]) {
        await expect(cards.filter({ has: page.getByRole("heading", { name, exact: true }) })
          .getByRole("button", { name: "Installed", exact: true })).toBeDisabled();
      }
      await expect(cards.filter({ has: page.getByRole("heading", { name: "Available create variant", exact: true }) })
        .getByRole("button", { name: "Install", exact: true })).toBeEnabled();
      await expect(rows).toHaveCount(50);

      const more = page.getByRole("button", { name: "More installed models", exact: true });
      await more.focus();
      await more.press("Enter");
      await expect.poll(() => Boolean(releasePage)).toBe(true);
      const loading = page.getByRole("button", { name: "Loading installed models…", exact: true });
      await expect(loading).toBeFocused();
      await expect(loading).toHaveAttribute("aria-disabled", "true");
      releasePage?.();
      await expect(page.getByText("Installed page temporarily unavailable", { exact: true })).toBeVisible();
      await expect(rows).toHaveCount(50);
      failNextPage = false;
      await page.getByRole("button", { name: "Retry installed models", exact: true }).click();
      await expect(rows).toHaveCount(56);
      await expect(page.getByText("Installed page 55 %_", { exact: true })).toBeVisible();
      await page.getByRole("textbox", { name: "Search installed models" }).fill("55 %_");
      await expect(rows).toHaveCount(1);
      await expect(page.getByRole("textbox", { name: "Search installed models" })).toBeFocused();
      await expect(cards.getByRole("button", { name: "Installed", exact: true })).toHaveCount(2);
      expect(reads.some(url => url.pathname === "/api/models" && url.searchParams.get("offset") === "50")).toBe(true);
      expect(reads.some(url => url.pathname === "/api/models" && url.searchParams.get("search") === "55 %_" && url.searchParams.get("offset") === "0")).toBe(true);
      expect(reads.filter(url => url.pathname === "/api/models").every(url => Number(url.searchParams.get("limit")) > 0 && Number(url.searchParams.get("limit")) <= 50)).toBe(true);
      expect(reads.filter(url => url.pathname === "/api/profiles").every(url => Number(url.searchParams.get("limit")) > 0 && Number(url.searchParams.get("limit")) <= 200)).toBe(true);
      expect(reads.some(url => url.pathname === "/api/profiles" && url.searchParams.get("limit") === "1" && url.searchParams.get("install_id") === "installed-page-55")).toBe(true);
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    } finally {
      releaseStatus();
      releasePage?.();
      await page.unrouteAll({ behavior: "ignoreErrors" });
      await page.goto("about:blank");
    }
  });
}
