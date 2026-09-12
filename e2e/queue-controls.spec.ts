import { expect, test, type Route } from "@playwright/test";
import type { QueueActivityItem } from "../apps/web/src/types";

const stamp = "2026-09-01T00:00:00Z";

function queuedPlan(id: string, title: string): QueueActivityItem {
  return {
    owner_type: "work_plan", owner_id: id, label: "Submitted work", lane: "generation",
    status: "queued", chat_id: "chat-" + id, chat_title: title,
    created_at: stamp, updated_at: stamp, step_count: 2, completed_steps: 0,
    blocked_steps: 0, active_jobs: 2, running_jobs: 0, queued_jobs: 2, paused_jobs: 0,
    progress: null, control_state: "eligible", control_revision: 0, allowed_actions: ["hold"],
  };
}

for (const firstToFinish of ["plan-a", "plan-b"]) {
  test(`queue requests preserve focus across two pending rows: ${firstToFinish} finishes first`, async ({ page }) => {
    const items = [
      queuedPlan("plan-a", "First landscape"),
      queuedPlan("plan-b", "Second landscape"),
    ];
    const waiting = new Map<string, Route>();
    const requests: string[] = [];
    await page.route("**/api/queue/activity?*", async (route) => {
      await route.fulfill({ json: {
        items, total: items.length,
        lane_counts: { generation: items.length, transfer: 0, install: 0 },
        next_cursor: null, observed_at: stamp,
      } });
    });
    await page.route("**/api/queue/items/*/hold", (route) => {
      const id = new URL(route.request().url()).pathname.split("/").at(-2)!;
      requests.push(id);
      waiting.set(id, route);
    });
    async function finish(id: string) {
      const item = items.find((entry) => entry.owner_id === id)!;
      item.control_state = "held";
      item.control_revision = 1;
      item.allowed_actions = ["release"];
      const route = waiting.get(id)!;
      await route.fulfill({ json: {
        owner_id: id, control_state: "held", control_revision: 1, eligible_since: null,
      } });
      waiting.delete(id);
    }
    await page.route("**/api/queue/lanes/generation", (route) => route.fulfill({ json: {
      lane: "generation", dispatch_state: "open", revision: 0,
      running_jobs: 0, allowed_actions: ["pause_after_current"],
    } }));
    await page.goto("/");
    const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
    await expect(setup).toBeVisible();
    await setup.getByRole("button", { name: "Not now" }).click();
    await page.getByRole("button", { name: "View accepted work", exact: true }).click();
    const first = page.getByRole("button", { name: /^(Hold|Saving hold for) First landscape$/ });
    const second = page.getByRole("button", { name: /^(Hold|Saving hold for) Second landscape$/ });
    await first.focus();
    await page.keyboard.press("Enter");
    await expect.poll(() => waiting.has("plan-a")).toBe(true);
    await expect(first).toHaveText("Saving…");
    await expect(first).toBeFocused();
    // A keyboard repeat is handled by the production pending guard.
    await page.keyboard.press("Enter");
    expect(requests).toEqual(["plan-a"]);
    await page.keyboard.press("Tab");
    await expect(page.getByRole("button", { name: "Show steps for First landscape" })).toBeFocused();
    await page.keyboard.press("Tab");
    await expect(second).toBeFocused();
    await page.keyboard.press("Enter");
    await expect.poll(() => waiting.has("plan-b")).toBe(true);
    await expect(second).toHaveText("Saving…");
    await expect(second).toBeFocused();
    await finish(firstToFinish);
    const firstFinishedTitle = firstToFinish === "plan-a" ? "First landscape" : "Second landscape";
    await expect(page.getByRole("button", { name: "Release " + firstFinishedTitle })).toBeVisible();
    const focusedSecond = page.getByRole("button", {
      name: (firstToFinish === "plan-b" ? "Release " : "Saving hold for ") + "Second landscape",
      exact: true,
    });
    await expect(focusedSecond).toBeFocused();
    await finish(firstToFinish === "plan-a" ? "plan-b" : "plan-a");
    const releasedSecond = page.getByRole("button", { name: "Release Second landscape", exact: true });
    await expect(releasedSecond).toBeVisible();
    await expect(releasedSecond).toBeFocused();
    expect(requests).toEqual(["plan-a", "plan-b"]);
    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog", { name: "Accepted work", exact: true })).toBeHidden();
  });
}
