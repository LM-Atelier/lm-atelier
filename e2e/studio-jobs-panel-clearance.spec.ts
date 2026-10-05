import { permanentlyDeleteChat } from "./recovery-cleanup";
import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

/** The jobs panel beside Image Studio's controls, measured in a real browser.
 *
 * The panel floats at the window's bottom right while work runs or a job has
 * failed, which is where Image Studio keeps its options and main action. What
 * would lie under a floating panel changes with its own height, the open tool
 * and the length of the edit history, so the Studio keeps the panel wholly
 * above it. Only a real browser lays either out.
 */

async function csrf(request: APIRequestContext): Promise<string> {
  const response = await request.post("/api/session");
  expect(response.ok()).toBeTruthy();
  return ((await response.json()) as { csrf_token: string }).csrf_token;
}

/** A plain opaque picture of its own, so it opens a Studio session of its own. */
async function picture(page: Page, width: number): Promise<Buffer> {
  const dataUrl = await page.evaluate((seed) => {
    const canvas = document.createElement("canvas");
    canvas.width = 120;
    canvas.height = 80;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no canvas");
    context.fillStyle = "#d9d4c7";
    context.fillRect(0, 0, 120, 80);
    context.fillStyle = "#3a6ea5";
    context.fillRect(10, 30, 30, 30);
    context.fillStyle = `rgb(${seed % 256}, 160, 90)`;
    context.beginPath();
    context.arc(85, 45, 16, 0, Math.PI * 2);
    context.fill();
    return canvas.toDataURL("image/png");
  }, width);
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

const VIEWPORTS = [
  { width: 1280, height: 800 },
  { width: 1440, height: 900 },
  { width: 1680, height: 1000 },
];

for (const viewport of VIEWPORTS) {
  test(`at ${viewport.width} pixels the jobs panel sits above Image Studio while work waits`, async ({ browser, request }) => {
    const headers = { "x-local-lm-csrf": await csrf(request) };
    // Work held in the queue keeps the panel up for the whole measurement.
    const lane = (await (await request.get("/api/queue/lanes/generation")).json()) as { revision: number };
    const paused = await request.post("/api/queue/lanes/generation/pause-after-current", {
      headers, data: { expected_revision: lane.revision, idempotency_key: `pause-studio-clearance-${viewport.width}` },
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
      await context.addInitScript(() => sessionStorage.setItem("lm-atelier-setup-dismissed", "1"));
      const page = await context.newPage();
      await page.goto("/?view=studio");
      await page.getByLabel("Choose an image to edit").setInputFiles(
        { name: `held-${viewport.width}.png`, mimeType: "image/png", buffer: await picture(page, viewport.width) },
      );
      const studio = page.locator(".studio-view");
      await expect(studio.getByRole("button", { name: "Adjust light and color" })).toBeVisible();
      const panel = page.locator(".jobs-panel");
      await expect(panel).toBeVisible();

      // The panel ends above the Studio's top edge, so no part of the Studio is under it.
      const panelBox = await panel.boundingBox();
      const studioBox = await studio.boundingBox();
      expect(panelBox).not.toBeNull();
      expect(studioBox).not.toBeNull();
      expect(panelBox!.y + panelBox!.height).toBeLessThanOrEqual(studioBox!.y);
      // And the Studio still has its whole rail and options below it.
      await expect(studio.getByRole("navigation", { name: "Editing tools" })).toBeInViewport({ ratio: 1 });
      await expect(studio.locator(".studio-panel")).toBeInViewport({ ratio: 1 });
    } finally {
      await context.close();
      // Let the held work finish before its chat goes, so no cancelled job is
      // left showing for whatever runs next in this browser.
      const current = (await (await request.get("/api/queue/lanes/generation")).json()) as { revision: number };
      expect.soft((await request.post("/api/queue/lanes/generation/resume", {
        headers, data: { expected_revision: current.revision, idempotency_key: `resume-studio-clearance-${viewport.width}` },
      })).ok()).toBeTruthy();
      try {
        await expect.poll(async () => {
          const jobs = (await (await request.get("/api/jobs")).json()) as Array<{ run_id: string | null; status: string }>;
          return jobs.filter((job) => job.run_id === run.id).map((job) => job.status);
        }, { timeout: 30_000 }).toEqual(["complete"]);
      } finally {
        await permanentlyDeleteChat(request, id, headers);
      }
    }
  });
}
