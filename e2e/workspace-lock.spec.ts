import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

/** Leave the workspace unlocked with the lock off, since every spec shares one server. */
async function openWorkspaceForOthers(request: APIRequestContext) {
  const session = await request.post("/api/session");
  expect(session.ok()).toBeTruthy();
  const { csrf_token: token } = await session.json() as { csrf_token: string };
  const headers = { "x-local-lm-csrf": token };
  const unlocked = await request.post("/api/privacy/unlock", { headers, data: {} });
  expect(unlocked.ok()).toBeTruthy();
  const policy = await request.get("/api/privacy/policy");
  expect(policy.ok()).toBeTruthy();
  const { enabled, revision } = await policy.json() as { enabled: boolean; revision: number };
  if (!enabled) return;
  const off = await request.put("/api/privacy/policy", {
    headers, data: { expected_revision: revision, enabled: false },
  });
  expect(off.ok()).toBeTruthy();
}

function lockedHeading(page: Page) {
  return page.getByRole("heading", { name: "LM Atelier is locked" });
}

test.afterAll(async ({ request }) => {
  await openWorkspaceForOthers(request);
});

test("locks every open window from Settings, and one unlock opens them all", async ({ browser }) => {
  const context = await browser.newContext();
  await context.addInitScript(() => sessionStorage.setItem("lm-atelier-setup-dismissed", "1"));
  try {
    const first = await context.newPage();
    await first.goto("/?view=settings&settings=privacy");
    const settingHeading = first.getByRole("heading", { name: "Workspace lock" });
    await expect(settingHeading).toBeVisible();
    const on = first.getByRole("button", { name: "On", exact: true });
    await on.click();
    await expect(on).toHaveAttribute("aria-pressed", "true");
    await expect(first.getByText("Workspace lock turned on.")).toBeVisible();

    await first.getByRole("button", { name: "Lock now", exact: true }).click();
    await expect(lockedHeading(first)).toBeVisible();
    await expect(settingHeading).toHaveCount(0);

    const second = await context.newPage();
    let sessions = 0;
    second.on("request", (request) => {
      if (new URL(request.url()).pathname === "/api/session") sessions += 1;
    });
    await second.goto("/");
    await expect(lockedHeading(second)).toBeVisible();
    await expect(second.getByRole("link", { name: "Skip to main content" })).toHaveCount(0);
    // A locked window waits quietly: no session renewed every second, no
    // live connection retried in a loop.
    await second.waitForTimeout(5_000);
    expect(sessions).toBeLessThanOrEqual(2);

    await second.getByRole("button", { name: "Unlock", exact: true }).click();
    await expect(lockedHeading(second)).toHaveCount(0);
    await expect(second.getByRole("link", { name: "Skip to main content" })).toBeAttached();

    // The first window learns of the unlock on its own, within one status poll.
    await expect(lockedHeading(first)).toHaveCount(0, { timeout: 10_000 });
    await expect(settingHeading).toBeVisible();
  } finally {
    await context.close();
  }
});
