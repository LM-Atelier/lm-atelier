import { expect, test } from "@playwright/test";

for (const width of [1280, 375]) {
  test(`browse and edit profiles and presets beyond the first page at ${width}px`, async ({ page, request }) => {
    test.setTimeout(120_000);
    await page.setViewportSize({ width, height: 812 });
    const { csrf_token } = await (await request.post("/api/session")).json() as { csrf_token: string };
    const headers = { "x-local-lm-csrf": csrf_token };
    const created: Array<{ resource: string; id: string }> = [];
    const reads: URL[] = [];
    const prefix = `Page study ${width}`;
    let failPresetPage = true;
    let holdProfilePage = true;
    let releasePage: (() => void) | undefined;
    try {
      for (const resource of ["profiles", "presets"]) {
        for (let index = 0; index < 56; index++) {
          const name = `${prefix} ${String(index).padStart(2, "0")}${index === 55 ? " Straße %_" : ""}`;
          const response = await request.post(`/api/${resource}`, {
            headers,
            data: { name, role: "chat", ...(resource === "profiles" ? { engine: "mock" } : {}) },
          });
          expect(response.status(), await response.text()).toBe(201);
          created.push({ resource, id: (await response.json() as { id: string }).id });
        }
      }
      await page.addInitScript(() => sessionStorage.setItem("lm-atelier-setup-dismissed", "1"));
      page.on("request", (message) => {
        const url = new URL(message.url());
        if (["/api/profiles", "/api/presets"].includes(url.pathname) && message.method() === "GET") reads.push(url);
      });
      await page.route("**/api/profiles?*", async (route) => {
        const url = new URL(route.request().url());
        if (holdProfilePage && url.searchParams.get("offset") === "50") {
          holdProfilePage = false;
          await new Promise<void>((resolve) => { releasePage = resolve; });
        }
        await route.continue();
      });
      await page.route("**/api/presets?*", async (route) => {
        const url = new URL(route.request().url());
        if (failPresetPage && url.searchParams.get("offset") === "50") {
          await route.fulfill({ status: 503, json: { detail: "Preset choices temporarily unavailable" } });
        } else await route.continue();
      });
      await page.goto("/?view=settings&settings=models-and-generation");
      for (const resource of ["profiles", "presets"]) {
        const profile = resource === "profiles";
        const label = profile ? "model profiles" : "generation presets";
        const noun = profile ? "profile" : "preset";
        const section = page.locator("section").filter({
          has: page.getByRole("heading", { name: profile ? "Model profiles" : "Generation presets", exact: true }),
        });
        const search = section.getByRole("searchbox", { name: `Search ${label}` });
        await search.fill(prefix);
        const editors = section.getByRole("button", { name: new RegExp(`^Edit ${noun}: ${prefix}`) });
        await expect(editors).toHaveCount(50);
        const more = section.getByRole("button", { name: `More ${label}`, exact: true });
        await more.focus();
        await page.keyboard.press("Enter");
        if (profile) {
          await expect.poll(() => Boolean(releasePage)).toBe(true);
          await expect(section.getByRole("button", { name: `Loading ${label}...`, exact: true })).toBeFocused();
          releasePage?.();
          releasePage = undefined;
        } else {
          await expect(section.getByText("Preset choices temporarily unavailable", { exact: true })).toBeVisible();
          await expect(editors).toHaveCount(50);
          failPresetPage = false;
          await section.getByRole("button", { name: `Retry ${label}` }).click();
        }
        await expect(editors).toHaveCount(56);
        const original = `${prefix} 55 Straße %_`;
        await section.getByRole("button", { name: `Edit ${noun}: ${original}`, exact: true }).click();
        const dialog = page.getByRole("dialog", { name: `Edit ${noun}`, exact: true });
        await expect(dialog.getByLabel(profile ? "Profile name" : "Preset name", { exact: true })).toHaveValue(original);
        const renamed = `${original} kept`;
        await dialog.getByLabel(profile ? "Profile name" : "Preset name", { exact: true }).fill(renamed);
        await dialog.getByRole("button", { name: `Save ${noun}`, exact: true }).click();
        await expect(dialog).toHaveCount(0);
        await search.fill("STRASSE %_");
        await expect(editors).toHaveCount(1);
        await expect(section.getByRole("button", { name: `Edit ${noun}: ${renamed}`, exact: true })).toBeVisible();
        const persisted = await request.get(`/api/${resource}?search=${encodeURIComponent(renamed)}&limit=50`);
        expect((await persisted.json() as Array<{ name: string }>).map((row) => row.name)).toEqual([renamed]);
      }
      expect(reads.length).toBeGreaterThan(4);
      expect(reads.every((url) => Number(url.searchParams.get("limit")) > 0 && Number(url.searchParams.get("limit")) <= 200)).toBe(true);
      for (const resource of ["profiles", "presets"]) {
        expect(reads.some((url) => url.pathname === `/api/${resource}` && url.searchParams.get("offset") === "50")).toBe(true);
        expect(reads.some((url) => url.pathname === `/api/${resource}` && url.searchParams.get("search") === "STRASSE %_" && url.searchParams.get("offset") === "0")).toBe(true);
      }
    } finally {
      releasePage?.();
      await page.unrouteAll({ behavior: "ignoreErrors" });
      await page.goto("about:blank");
      for (const item of created) {
        expect.soft((await request.delete(`/api/${item.resource}/${item.id}`, { headers })).ok()).toBe(true);
      }
    }
  });
}
