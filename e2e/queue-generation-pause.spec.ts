import { expect, test, type Route } from "@playwright/test";
import type { QueueActivityItem, QueueControlCommand } from "../apps/web/src/types";

for (const width of [1280, 390]) {
  test("generation pause preserves keyboard control and recovers missed events at " + width, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 900 });
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.clock.install();
    let current = {
      lane: "generation", dispatch_state: "open", revision: 0,
      running_jobs: 1, allowed_actions: ["pause_after_current"],
    };
    const item: QueueActivityItem = {
      owner_type: "work_plan", owner_id: "example-plan", label: "Submitted work", lane: "generation",
      status: "running", chat_id: "example-chat", chat_title: "Example generation",
      created_at: "2026-09-01T00:00:00Z", updated_at: "2026-09-01T00:00:00Z",
      step_count: 1, completed_steps: 0, blocked_steps: 0, active_jobs: 1,
      running_jobs: 1, queued_jobs: 0, paused_jobs: 0, progress: null,
      control_state: "eligible", control_revision: 0, allowed_actions: [],
    };
    const requests: Array<{ action: string; command: QueueControlCommand }> = [];
    let pending: Route | undefined;
    await page.route("**/api/queue/activity?*", (route) => route.fulfill({ json: {
      items: current.running_jobs ? [item] : [], total: current.running_jobs,
      lane_counts: { generation: current.running_jobs, transfer: 0, install: 0 },
      next_cursor: null, observed_at: "2026-09-01T00:00:00Z",
    } }));
    await page.route("**/api/queue/lanes/generation", (route) => route.fulfill({ json: current }));
    await page.route("**/api/queue/lanes/generation/*", async (route) => {
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
    await page.getByRole("button", { name: "View accepted work", exact: true }).click();
    const pause = page.getByRole("button", { name: "Pause generation after current work", exact: true });
    await pause.focus();
    await page.keyboard.press("Enter");
    await expect.poll(() => pending !== undefined).toBe(true);
    const saving = page.getByRole("button", { name: "Saving generation change", exact: true });
    await expect(saving).toBeFocused();
    await expect(saving).toBeDisabled();
    await page.keyboard.press("Enter");
    expect(requests).toHaveLength(1);
    expect(requests[0].command.expected_revision).toBe(0);
    await page.getByRole("combobox", { name: "Work category" }).focus();
    current = { ...current, dispatch_state: "draining", revision: 1, allowed_actions: ["resume"] };
    await pending!.fulfill({ json: current });
    await expect(page.getByText("Finishing current generation. New generation will wait.")).toBeVisible();
    await expect(page.getByRole("combobox", { name: "Work category" })).toBeFocused();
    current = { ...current, dispatch_state: "paused", revision: 2, running_jobs: 0 };
    await page.clock.fastForward(5_100);
    await expect(page.getByText("Generation paused. New submissions stay queued.")).toBeVisible();
    await expect(page.getByText("No active accepted work in this category.")).toBeVisible();
    const overflow = await page.evaluate(() => ({
      document: document.documentElement.scrollWidth > window.innerWidth,
      dialog: (() => {
        const element = document.querySelector<HTMLElement>(".queue-activity-dialog")!;
        return element.scrollWidth > element.clientWidth;
      })(),
    }));
    expect(overflow).toEqual({ document: false, dialog: false });
    await page.screenshot({ path: testInfo.outputPath("generation-paused.png"), fullPage: true });
    const resume = page.getByRole("button", { name: "Resume generation", exact: true });
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
    await expect(page.getByRole("button", { name: "View accepted work", exact: true })).toBeFocused();
  });
}
