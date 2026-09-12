import { expect, test, type Route } from "@playwright/test";

for (const cause of ["manual", "automatic"] as const) {
  test("queue refresh keeps keyboard focus during " + cause + " requests", async ({ page }) => {
    let requests = 0;
    let pending: Route | undefined;
    const result = {
      items: [], total: 0, next_cursor: null,
      lane_counts: { generation: 0, transfer: 0, install: 0 },
      observed_at: "2026-09-01T00:00:00Z",
    };
    await page.clock.install();
    await page.route("**/api/queue/activity?*", async (route) => {
      requests += 1;
      if (requests === 1) await route.fulfill({ json: result });
      else pending = route;
    });
    await page.route("**/api/queue/lanes/generation", (route) => route.fulfill({ json: {
      lane: "generation", dispatch_state: "open", revision: 0,
      running_jobs: 0, allowed_actions: ["pause_after_current"],
    } }));
    await page.goto("/");
    const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
    await expect(setup).toBeVisible();
    await setup.getByRole("button", { name: "Not now" }).click();
    await page.getByRole("button", { name: "View accepted work", exact: true }).click();
    await expect(page.getByText("No active accepted work in this category.")).toBeVisible();
    const refresh = page.getByRole("button", { name: /^Refresh(?:ing)? accepted work$/ });
    await page.getByRole("combobox", { name: "Work category" }).focus();
    await page.keyboard.press("Tab");
    await expect(refresh).toBeFocused();
    if (cause === "manual") await page.keyboard.press("Enter");
    else await page.clock.fastForward(5_000);
    await expect.poll(() => requests).toBe(2);
    await expect.poll(() => pending !== undefined).toBe(true);
    await expect(refresh).toBeDisabled();
    await expect(refresh).toBeFocused();
    await page.keyboard.press("Enter");
    await page.keyboard.press("Enter");
    expect(requests).toBe(2);
    await pending!.fulfill({ json: result });
    await expect(page.getByText("No active accepted work in this category.")).toBeVisible();
    await expect(refresh).toBeFocused();
    expect(requests).toBe(2);
    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog", { name: "Accepted work", exact: true })).toBeHidden();
    await expect(page.getByRole("button", { name: "View accepted work", exact: true })).toBeFocused();
  });
}
