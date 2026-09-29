import { expect, test } from "@playwright/test";
import type { QueueOrderPage } from "../apps/web/src/queueOrderTypes";

test("keyboard ordering persists through the real API and a page reload", async ({ page, request }) => {
  const session = await request.post("/api/session");
  expect(session.ok()).toBeTruthy();
  const { csrf_token: token } = await session.json() as { csrf_token: string };
  const headers = { "x-local-lm-csrf": token };
  const modelPath = process.env.LM_ATELIER_E2E_MODEL_PATH;
  expect(modelPath).toBeTruthy();
  const imported = await request.post("/api/models/import", {
    headers, data: { name: "Queue text fixture", role: "chat", engine: "mock", local_path: modelPath },
  });
  expect(imported.status()).toBe(201);
  const current = await request.get("/api/queue/lanes/generation");
  expect(current.ok()).toBeTruthy();
  const policy = await current.json() as { revision: number };
  const paused = await request.post("/api/queue/lanes/generation/pause-after-current", {
    headers, data: { expected_revision: policy.revision, idempotency_key: "pause-for-ordering" },
  });
  expect(paused.ok()).toBeTruthy();
  const chats: string[] = [];
  const runs: string[] = [];
  try {
    for (const name of ["Blue bowl", "Green cup", "Red vase"]) {
      const created = await request.post("/api/chats", {
        headers, data: { title: name, routing_mode: "text" },
      });
      expect(created.status()).toBe(201);
      const { id } = await created.json() as { id: string };
      chats.push(id);
      const accepted = await request.post(`/api/chats/${id}/turns`, {
        headers, data: { text: `Describe a ${name.toLowerCase()} in one sentence.`, mode: "text" },
      });
      expect(accepted.status()).toBe(202);
      const { run } = await accepted.json() as { run: { id: string } };
      runs.push(run.id);
    }
    const response = await request.get("/api/queue/lanes/generation/order");
    expect(response.ok()).toBeTruthy();
    const before = await response.json() as QueueOrderPage;
    expect(before.items).toHaveLength(3);
    expect(before.items.every((item) => item.cohort_id === before.items[0].cohort_id)).toBe(true);
    const moved = before.items[2];
    const expected = [before.items[0].owner, moved.owner, before.items[1].owner];
    await page.goto("/");
    const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
    await expect(setup).toBeVisible();
    await setup.getByRole("button", { name: "Not now" }).click();
    await page.getByRole("button", { name: "Accepted work", exact: true }).click();
    await page.getByRole("button", { name: "Change dispatch order" }).click();
    const dialog = page.getByRole("dialog", { name: "Dispatch order", exact: true });
    const earlier = dialog.getByRole("listitem", { name: moved.label })
      .getByRole("button", { name: "Move earlier" });
    await earlier.focus();
    await page.keyboard.press("Enter");
    await expect(dialog.getByText(/Order saved/)).toBeVisible();
    await expect(earlier).toBeFocused();
    const saved = await request.get("/api/queue/lanes/generation/order");
    expect(saved.ok()).toBeTruthy();
    expect((await saved.json() as QueueOrderPage).items.map((item) => item.owner)).toEqual(expected);
    await page.reload();
    await page.getByRole("button", { name: "Accepted work", exact: true }).click();
    await page.getByRole("button", { name: "Change dispatch order" }).click();
    await expect(dialog.getByRole("listitem").nth(1)).toHaveAccessibleName(moved.label);
    const stillPaused = await request.get("/api/queue/lanes/generation");
    expect((await stillPaused.json() as { dispatch_state: string }).dispatch_state).toBe("paused");
  } finally {
    await page.goto("about:blank");
    const latest = await request.get("/api/queue/lanes/generation");
    const { revision } = await latest.json() as { revision: number };
    expect.soft((await request.post("/api/queue/lanes/generation/resume", {
      headers, data: { expected_revision: revision, idempotency_key: "resume-after-ordering" },
    })).ok()).toBeTruthy();
    try {
      await expect.poll(async () => {
        const response = await request.get("/api/jobs");
        expect(response.ok()).toBeTruthy();
        const jobs = await response.json() as Array<{ run_id: string | null; status: string }>;
        return jobs.filter((job) => job.run_id !== null && runs.includes(job.run_id))
          .map((job) => job.status);
      }, { timeout: 30_000 }).toEqual(runs.map(() => "complete"));
    } finally {
      for (const id of chats) expect.soft((await request.delete(`/api/chats/${id}`, { headers })).ok()).toBeTruthy();
    }
  }
});
