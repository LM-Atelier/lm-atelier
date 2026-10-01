import { expect, test } from "@playwright/test";
import type { Job, ModelInstall, ModelProfile } from "../apps/web/src/types";

const stamp = "2026-09-30T00:00:00Z";
const replacement: ModelInstall = {
  id: "updated-model", source_id: null, name: "Updated landscape model", role: "image", engine: "comfyui",
  local_path: "neutral/replacement", size_bytes: 100, compatibility: "likely", manifest_json: {},
  active: true, readiness: "ready", capability_evidence: null, created_at: stamp, updated_at: stamp,
};
const job: Job = {
  id: "profile-page-update", kind: "download", status: "complete", run_id: null,
  progress: 1, phase: "complete", payload_json: {}, result_json: { model_install_id: replacement.id },
  error: null, attempt: 1, cancellable: false, created_at: stamp, updated_at: stamp,
  started_at: stamp, completed_at: stamp,
};

for (const width of [1280, 375]) {
  test(`switches an update profile beyond the first page at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 812 });
    const profiles: ModelProfile[] = Array.from({ length: 51 }, (_, index) => ({
      id: `update-profile-${index}`, model_install_id: "previous-model", name: `Update profile ${index}`,
      use_case: "", role: "image", engine: "comfyui", load_settings_json: {}, request_settings_json: {},
      is_default: false,
    }));
    const reads: URL[] = [];
    const switches: unknown[] = [];
    let failPage = true;
    let holdPage = true;
    let releasePage: (() => void) | undefined;
    let releaseSwitch: (() => void) | undefined;
    page.on("request", request => {
      const url = new URL(request.url());
      if (request.method() === "GET" && ["/api/models", "/api/profiles"].includes(url.pathname)) reads.push(url);
    });
    await page.route(/\/api\/models(?:\?|$)/, route => route.fulfill({ json:
      new URL(route.request().url()).searchParams.get("install_id") === replacement.id ? [replacement] : [],
    }));
    await page.route(`**/api/downloads/${job.id}`, route => route.fulfill({ json: job }));
    await page.route(/\/api\/profiles(?:\?|$)/, async route => {
      const query = new URL(route.request().url()).searchParams;
      if (query.has("limit") && query.get("install_id") !== "previous-model") return route.fulfill({ json: [] });
      const offset = Number(query.get("offset") ?? 0);
      if (offset === 50) {
        if (holdPage) {
          holdPage = false;
          await new Promise<void>(resolve => { releasePage = resolve; });
        }
        if (failPage) return route.fulfill({ status: 503, json: { detail: "Update profiles temporarily unavailable" } });
      }
      const eligible = profiles.filter(profile => profile.model_install_id === "previous-model");
      await route.fulfill({ json: eligible.slice(offset, offset + Number(query.get("limit") ?? eligible.length)) });
    });
    await page.route("**/api/profiles/update-profile-50/model-update", async route => {
      switches.push(route.request().postDataJSON());
      await new Promise<void>(resolve => { releaseSwitch = resolve; });
      profiles[50].model_install_id = replacement.id;
      await route.fulfill({ json: profiles[50] });
    });
    await page.route(/\/api\/catalog\?/, route => route.fulfill({ json: { items: [], next_cursor: null, stale: false } }));
    await page.addInitScript(() => {
      sessionStorage.setItem("lm-atelier-setup-dismissed", "1");
      localStorage.setItem("lm-atelier.model-update-downloads", JSON.stringify([
        { jobId: "profile-page-update", previousInstallId: "previous-model", modelName: "Landscape update" },
      ]));
    });
    try {
      await page.goto("/?view=models");
      const offer = page.getByRole("region", { name: "Update Landscape update", exact: true });
      const choices = offer.getByRole("button", { name: /^Switch Update profile / });
      await expect(choices).toHaveCount(50);
      await expect(offer.getByRole("button", { name: "Switch Update profile 50", exact: true })).toHaveCount(0);
      const more = offer.getByRole("button", { name: "More profiles", exact: true });
      await more.focus();
      await more.press("Enter");
      await expect.poll(() => Boolean(releasePage)).toBe(true);
      await expect(offer.getByRole("button", { name: "Loading profiles...", exact: true })).toBeFocused();
      releasePage?.();
      await expect(offer.getByText("Update profiles temporarily unavailable", { exact: true })).toBeVisible();
      await expect(choices).toHaveCount(50);
      await expect(offer.getByText("No profiles use the previous version.", { exact: true })).toHaveCount(0);
      failPage = false;
      await offer.getByRole("button", { name: "Retry profiles", exact: true }).click();
      await expect(choices).toHaveCount(51);
      const chosen = offer.getByRole("button", { name: "Switch Update profile 50", exact: true });
      await chosen.focus();
      await chosen.press("Enter");
      await expect.poll(() => Boolean(releaseSwitch)).toBe(true);
      await expect(chosen).toBeFocused();
      await expect(chosen).toHaveAttribute("aria-disabled", "true");
      expect(switches).toEqual([{ expected_install_id: "previous-model", download_job_id: job.id }]);
      releaseSwitch?.();
      await expect(offer.getByText("Updated Update profile 50.", { exact: true })).toBeVisible();
      await expect(chosen).toHaveCount(0);
      await expect(offer.getByRole("button", { name: "Switch Update profile 0", exact: true })).toBeVisible();
      const offerReads = reads.filter(url => url.pathname === "/api/profiles" && url.searchParams.get("install_id") === "previous-model");
      expect(offerReads.some(url => url.searchParams.get("offset") === "50")).toBe(true);
      expect(offerReads.every(url => url.searchParams.get("limit") === "50"
        && url.searchParams.get("role") === "image" && url.searchParams.get("engine") === "comfyui")).toBe(true);
      expect(reads.filter(url => url.pathname === "/api/profiles").every(url => Number(url.searchParams.get("limit")) > 0)).toBe(true);
      expect(reads.some(url => url.pathname === "/api/models" && url.searchParams.get("install_id") === replacement.id)).toBe(true);
      await offer.getByRole("button", { name: "Dismiss update", exact: true }).click();
      await expect(offer).toHaveCount(0);
      expect(await page.evaluate(() => JSON.parse(localStorage.getItem("lm-atelier.model-update-downloads") ?? "null"))).toEqual([]);
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    } finally {
      releasePage?.();
      releaseSwitch?.();
      await page.unrouteAll({ behavior: "ignoreErrors" });
      await page.goto("about:blank");
    }
  });
}
