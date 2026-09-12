import { expect, test } from "@playwright/test";

for (const width of [1280, 375]) {
  test("workspace and Settings history preserves destination and focus at " + width + "px", async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.route("**/api/setup/readiness", (route) => route.fulfill({
      json: { version: 2, state: "ready", roles: [] },
    }));
    await page.goto("/?view=settings&settings=data-and-backups&keep=1#anchor");
    const backups = page.getByRole("region", { name: "Data & backups" });
    await expect(backups).toBeVisible();
    await expect(page.locator("body")).toBeFocused();
    await page.evaluate(() => history.replaceState({ launcher: "neutral-fixture" }, "", location.href));
    const advanced = page.getByRole("region", { name: "Advanced" });
    const chooseAdvanced = async () => {
      if (width < 700) await page.getByRole("combobox", { name: "Settings section" }).selectOption("advanced");
      else await page.getByRole("button", { name: "Advanced", exact: true }).click();
    };
    await chooseAdvanced();
    await expect(advanced).toBeFocused();
    const length = await page.evaluate(() => history.length);
    await chooseAdvanced();
    await expect(advanced).toBeFocused();
    expect(await page.evaluate(() => history.length)).toBe(length);
    expect(await page.evaluate(() => history.state)).toEqual({ launcher: "neutral-fixture" });
    expect(new URL(page.url()).searchParams.get("keep")).toBe("1");
    expect(new URL(page.url()).hash).toBe("#anchor");
    if (width < 700) await page.getByRole("button", { name: "Toggle navigation" }).click();
    await page.getByRole("button", { name: "Workflows", exact: true }).click();
    await expect(page.locator("#main-content")).toBeFocused();
    await page.goBack();
    await expect(advanced).toBeFocused();
    await page.goBack();
    await expect(backups).toBeFocused();
    await page.goForward();
    await expect(advanced).toBeFocused();
    await page.goForward();
    await expect(page.locator("#main-content")).toBeFocused();
    expect(new URL(page.url()).searchParams.get("view")).toBe("workflows");
    expect(new URL(page.url()).searchParams.has("settings")).toBe(false);
    await page.goBack();
    await expect(advanced).toBeFocused();
    await page.reload();
    await expect(advanced).toBeVisible();
    await expect(page.locator("body")).toBeFocused();
  });
}
