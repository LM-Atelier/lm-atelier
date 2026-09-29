import { expect, test, type APIRequestContext } from "@playwright/test";

/** The jobs panel beside the composer, measured in a real browser.
 *
 * The panel floats over the page while work runs, and the composer floats at
 * the foot of the chat. jsdom lays nothing out, so whether one covers the
 * other can only be measured here, and at the narrow widths where it did.
 */

async function csrf(request: APIRequestContext): Promise<string> {
  const response = await request.post("/api/session");
  expect(response.ok()).toBeTruthy();
  return ((await response.json()) as { csrf_token: string }).csrf_token;
}

const VIEWPORTS = [
  { width: 360, height: 780 },
  { width: 390, height: 844 },
  { width: 1280, height: 800 },
];

for (const viewport of VIEWPORTS) {
  test(`at ${viewport.width} pixels the whole composer stays usable while work waits`, async ({ browser, request }) => {
    const headers = { "x-local-lm-csrf": await csrf(request) };
    // Work held in the queue keeps the panel up for the whole measurement.
    const lane = (await (await request.get("/api/queue/lanes/generation")).json()) as { revision: number };
    const paused = await request.post("/api/queue/lanes/generation/pause-after-current", {
      headers, data: { expected_revision: lane.revision, idempotency_key: `pause-clearance-${viewport.width}` },
    });
    expect(paused.ok()).toBeTruthy();
    const created = await request.post("/api/chats", { headers, data: { title: "Held work", routing_mode: "image" } });
    expect(created.status()).toBe(201);
    const { id } = (await created.json()) as { id: string };
    const accepted = await request.post(`/api/chats/${id}/turns`, {
      headers, data: { text: "A blue bowl on a wooden table", mode: "image" },
    });
    expect(accepted.status()).toBe(202);
    const { run } = (await accepted.json()) as { run: { id: string } };
    const context = await browser.newContext({ viewport });
    try {
      await context.addInitScript((current) => localStorage.setItem("local-lm-chat", current), id);
      const page = await context.newPage();
      await page.goto("/?view=chat");
      const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
      await expect(setup).toBeVisible();
      await setup.getByRole("button", { name: "Not now" }).click();
      const panel = page.locator(".jobs-panel");
      await expect(panel).toBeVisible();
      const composer = page.locator(".composer").first();
      await expect(composer).toBeVisible();
      const panelBox = await panel.boundingBox();
      const composerBox = await composer.boundingBox();
      expect(panelBox).not.toBeNull();
      expect(composerBox).not.toBeNull();
      // Above the composer's top edge, so no part of it is under the panel.
      expect(panelBox!.y + panelBox!.height).toBeLessThanOrEqual(composerBox!.y);

      // A real click on Attach reaches it: the file chooser opens.
      const chooser = page.waitForEvent("filechooser");
      await page.getByRole("button", { name: "Attach file" }).click();
      await chooser;
      // And Send is what a pointer finds at its middle, not the panel.
      const reached = await page.getByRole("button", { name: "Send" }).evaluate((button) => {
        const box = button.getBoundingClientRect();
        const found = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
        return found !== null && button.contains(found);
      });
      expect(reached).toBe(true);
    } finally {
      await context.close();
      // Let the held work finish before its chat goes, so no cancelled job is
      // left showing for whatever runs next in this browser.
      const current = (await (await request.get("/api/queue/lanes/generation")).json()) as { revision: number };
      expect.soft((await request.post("/api/queue/lanes/generation/resume", {
        headers, data: { expected_revision: current.revision, idempotency_key: `resume-clearance-${viewport.width}` },
      })).ok()).toBeTruthy();
      try {
        await expect.poll(async () => {
          const jobs = (await (await request.get("/api/jobs")).json()) as Array<{ run_id: string | null; status: string }>;
          return jobs.filter((job) => job.run_id === run.id).map((job) => job.status);
        }, { timeout: 30_000 }).toEqual(["complete"]);
      } finally {
        expect.soft((await request.delete(`/api/chats/${id}`, { headers })).ok()).toBeTruthy();
      }
    }
  });
}
