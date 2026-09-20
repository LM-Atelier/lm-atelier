import { expect, test, type Route } from "@playwright/test";
import type { QueueActivityItem, QueueControlCommand } from "../apps/web/src/types";

for (const width of [1280, 390]) {
  test("transfer pause preserves keyboard control and recovers missed events at " + width, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 900 });
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.clock.install();
    let current = {
      lane: "transfer", dispatch_state: "open", revision: 0,
      running_jobs: 1, allowed_actions: ["pause_after_current"],
    };
    const item: QueueActivityItem = {
      owner_type: "job", owner_id: "example-transfer", label: "Download", lane: "transfer",
      status: "running", chat_id: null, chat_title: null,
      created_at: "2026-09-01T00:00:00Z", updated_at: "2026-09-01T00:00:00Z",
      step_count: 0, completed_steps: 0, blocked_steps: 0, active_jobs: 1,
      running_jobs: 1, queued_jobs: 0, paused_jobs: 0, progress: null,
      control_state: "eligible", control_revision: 0, allowed_actions: [],
    };
    const requests: Array<{ action: string; command: QueueControlCommand }> = [];
    let pending: Route | undefined;
    await page.route("**/api/queue/activity?*", (route) => route.fulfill({ json: {
      items: current.running_jobs ? [item] : [], total: current.running_jobs,
      lane_counts: { generation: 0, transfer: current.running_jobs, install: 0 },
      next_cursor: null, observed_at: "2026-09-01T00:00:00Z",
    } }));
    await page.route("**/api/queue/lanes/transfer", (route) => route.fulfill({ json: current }));
    await page.route("**/api/queue/lanes/transfer/*", async (route) => {
      const action = new URL(route.request().url()).pathname.split("/").at(-1)!;
      requests.push({ action, command: route.request().postDataJSON() as QueueControlCommand });
      if (action === "pause-after-current") pending = route;
      else {
        current = { ...current, dispatch_state: "open", revision: 3, allowed_actions: ["pause_after_current"] };
        await route.fulfill({ json: current });
      }
    });
    await page.goto("/");
    const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
    await expect(setup).toBeVisible();
    await setup.getByRole("button", { name: "Not now" }).click();
    // At phone width accepted work sits in the sidebar, behind the navigation menu.
    const menu = page.getByRole("button", { name: "Toggle navigation" });
    if (await menu.isVisible()) await menu.click();
    await page.getByRole("button", { name: "Accepted work", exact: true }).click();
    const pause = page.getByRole("button", { name: "Pause transfers after current work", exact: true });
    await pause.focus();
    await page.keyboard.press("Enter");
    await expect.poll(() => pending !== undefined).toBe(true);
    const saving = page.getByRole("button", { name: "Saving transfers change", exact: true });
    await expect(saving).toBeFocused();
    await expect(saving).toBeDisabled();
    await page.keyboard.press("Enter");
    await page.keyboard.press("Space");
    expect(requests).toHaveLength(1);
    expect(requests[0].command.expected_revision).toBe(0);
    await page.getByRole("combobox", { name: "Work category" }).focus();
    current = { ...current, dispatch_state: "draining", revision: 1, allowed_actions: ["resume"] };
    await pending!.fulfill({ json: current });
    await expect(page.getByText("Finishing current transfers. New transfers will wait.")).toBeVisible();
    await expect(page.getByRole("combobox", { name: "Work category" })).toBeFocused();
    current = { ...current, dispatch_state: "paused", revision: 2, running_jobs: 0 };
    await page.clock.fastForward(5_100);
    await expect(page.getByText("Transfers paused. New submissions stay queued.")).toBeVisible();
    await expect(page.getByText("No active accepted work in this category.")).toBeVisible();
    const overflow = await page.evaluate(() => ({
      document: document.documentElement.scrollWidth > window.innerWidth,
      dialog: (() => {
        const element = document.querySelector<HTMLElement>(".queue-activity-dialog")!;
        return element.scrollWidth > element.clientWidth;
      })(),
    }));
    expect(overflow).toEqual({ document: false, dialog: false });
    await page.screenshot({ path: testInfo.outputPath("transfer-paused.png"), fullPage: true });
    const resume = page.getByRole("button", { name: "Resume transfers", exact: true });
    await resume.focus();
    await page.keyboard.press("Enter");
    await expect(pause).toBeVisible();
    await expect(pause).toBeFocused();
    expect(requests).toHaveLength(2);
    expect(requests[1].action).toBe("resume");
    expect(requests[1].command.expected_revision).toBe(2);
    expect(requests[1].command.idempotency_key).not.toBe(requests[0].command.idempotency_key);
    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog", { name: "Accepted work", exact: true })).toBeHidden();
    await expect(page.getByRole("button", { name: "Accepted work", exact: true })).toBeFocused();
  });
}
