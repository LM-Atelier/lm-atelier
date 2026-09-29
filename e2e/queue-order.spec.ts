import { expect, test, type Page } from "@playwright/test";
import type { QueueOrderCommand, QueueOrderPage } from "../apps/web/src/queueOrderTypes";

async function openOrder(page: Page) {
  let revision = 1;
  const items = ["First transfer", "Second transfer", "Third transfer"].map((label, index) => ({
    label, owner: { type: "job" as const, id: "download-" + index },
  }));
  const commands: QueueOrderCommand[] = [];
  const snapshot = (): QueueOrderPage => {
    const neighbours = items.map((_, i) => ({ before: items[i - 1]?.owner ?? null, after: items[i + 1]?.owner ?? null }));
    return { lane: "transfer", revision, total: items.length, next_cursor: null,
      items: items.map((item, i) => ({ ...item, queued_at: "2026-09-01T00:00:00Z", priority: 0,
        cohort_id: String(revision).padStart(64, "0"), position: i + 1, cohort_length: items.length,
        neighbors: neighbours[i], before_neighbors: neighbours[i - 1] ?? null,
        after_neighbors: neighbours[i + 1] ?? null, unavailable_reason: null,
      })),
    };
  };
  await page.route("**/api/queue/activity?*", (route) => route.fulfill({ json: {
    items: [], total: 0, lane_counts: { generation: 0, transfer: 0, install: 0 },
    next_cursor: null, observed_at: "2026-09-01T00:00:00Z",
  } }));
  for (const lane of ["generation", "transfer"]) {
    await page.route("**/api/queue/lanes/" + lane, (route) => route.fulfill({ json: {
      lane, revision: 0, dispatch_state: "paused", running_jobs: 0, allowed_actions: ["resume"],
    } }));
  }
  await page.route("**/api/queue/lanes/transfer/order?*", (route) => route.fulfill({ json: snapshot() }));
  await page.route("**/api/queue/lanes/transfer/reorder", async (route) => {
    const command = route.request().postDataJSON() as QueueOrderCommand;
    commands.push(command);
    const before = snapshot();
    const moved = before.items.find((item) => item.owner.id === command.owner.id)!;
    const anchor = before.items.find((item) => item.owner.id === (command.before ?? command.after)?.id)!;
    expect(command.expected_revision).toBe(revision);
    expect(command.cohort_id).toBe(moved.cohort_id);
    expect(command.expected_item_neighbors).toEqual(moved.neighbors);
    expect(command.expected_anchor_neighbors).toEqual(anchor.neighbors);
    const [item] = items.splice(items.findIndex((value) => value.owner.id === command.owner.id), 1);
    const index = items.findIndex((value) => value.owner.id === anchor.owner.id);
    items.splice(index + (command.after ? 1 : 0), 0, item);
    revision += 1;
    await route.fulfill({ json: { lane: "transfer", revision, owner: command.owner } });
  });
  await page.goto("/");
  const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setup).toBeVisible();
  await setup.getByRole("button", { name: "Not now" }).click();
  const menu = page.getByRole("button", { name: "Toggle navigation" });
  if (await menu.isVisible()) await menu.click();
  await page.getByRole("button", { name: "Accepted work", exact: true }).click();
  await page.getByLabel("Work category").selectOption("transfer");
  await page.getByRole("button", { name: "Change dispatch order" }).click();
  const dialog = page.getByRole("dialog", { name: "Dispatch order", exact: true });
  await expect(dialog.getByRole("listitem")).toHaveCount(3);
  return { dialog, commands, removeLastItem: () => { items.pop(); revision += 1; } };
}

for (const mode of ["desktop", "narrow", "zoomed"] as const) {
  test(`dispatch ordering preserves keyboard focus and fits ${mode} layout`, async ({ page }) => {
    await page.setViewportSize(mode === "narrow" ? { width: 390, height: 844 } : { width: 1440, height: 1000 });
    await page.emulateMedia({ reducedMotion: "reduce" });
    const { dialog, commands } = await openOrder(page);
    if (mode === "zoomed") await page.evaluate(() => { document.documentElement.style.zoom = "2"; });
    const moved = dialog.getByRole("listitem", { name: "Third transfer" });
    const earlier = moved.getByRole("button", { name: "Move earlier" });
    await earlier.focus();
    await page.keyboard.press("Enter");
    await expect(dialog.getByText(/Order saved/)).toBeVisible();
    await expect(earlier).toBeFocused();
    await expect(dialog.getByRole("listitem").nth(1)).toHaveAccessibleName("Third transfer");
    expect(commands).toHaveLength(1);
    expect(commands[0].before?.id).toBe("download-1");
    await expect(earlier).toHaveAttribute("aria-disabled", "false");
    await page.keyboard.press("Tab");
    await expect(moved.getByRole("button", { name: "Move later" })).toBeFocused();
    const size = await dialog.evaluate((element) => ({ width: element.clientWidth, scroll: element.scrollWidth }));
    expect(size.scroll).toBeLessThanOrEqual(size.width + 1);
    await page.keyboard.press("Escape");
    await expect(dialog).toBeHidden();
  });
}

test("dragging a transfer to another row submits one relative move", async ({ page }) => {
  const { dialog, commands } = await openOrder(page);
  const first = dialog.getByRole("listitem", { name: "First transfer" });
  const handle = dialog.getByRole("button", { name: "Drag Third transfer" });
  await handle.dragTo(first, { targetPosition: { x: 20, y: 4 } });
  await expect(dialog.getByText(/Order saved/)).toBeVisible();
  await expect(dialog.getByRole("listitem").first()).toHaveAccessibleName("Third transfer");
  expect(commands).toHaveLength(1);
  expect(commands[0].before?.id).toBe("download-0");
  expect(commands[0].after).toBeNull();
});

test("refresh keeps keyboard focus reachable when current work disappears", async ({ page }) => {
  await page.clock.install();
  const { dialog, removeLastItem } = await openOrder(page);
  await dialog.getByRole("listitem", { name: "Third transfer" })
    .getByRole("button", { name: "Move earlier" }).focus();
  removeLastItem();
  await page.clock.fastForward(5_000);
  await expect(dialog.getByRole("listitem", { name: "Third transfer" })).toBeHidden();
  await expect(dialog.getByRole("button", { name: "Refresh dispatch order" })).toBeFocused();
});
