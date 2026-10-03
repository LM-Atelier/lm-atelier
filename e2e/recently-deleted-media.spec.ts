import { expect, test } from "@playwright/test";
import type { RecoveryImpact, RecoveryItem } from "../apps/web/src/recoveryTypes";

for (const width of [1280, 390]) {
  test(`restores library membership and permanently removes its pins at ${width}px`, async ({ page, request }) => {
    test.setTimeout(90_000);
    await page.setViewportSize({ width, height: 900 });
    await page.route("**/api/setup/readiness", (route) => route.fulfill({ json: { version: 2, state: "ready", roles: [] } }));
    const session = await request.post("/api/session");
    const { csrf_token: csrf } = await session.json() as { csrf_token: string };
    const headers = { "x-local-lm-csrf": csrf };
    const title = "Garden recovery.png";
    const uploaded = await request.post("/api/artifacts?kind=image", { headers, multipart: {
      file: { name: title, mimeType: "image/png", buffer: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a6V8AAAAASUVORK5CYII=", "base64") },
    } });
    expect(uploaded.status()).toBe(201);
    const artifact = await uploaded.json() as { id: string };
    expect((await request.patch(`/api/artifacts/${encodeURIComponent(artifact.id)}`, { headers, data: { favorite: true } })).status()).toBe(200);
    const visible = await request.get("/api/artifact-library");
    expect(visible.status()).toBe(200);
    const entries = await visible.json() as { items: { id: string; artifact_id: string; favorite: boolean }[] };
    const entry = entries.items.find((item) => item.artifact_id === artifact.id);
    expect(entry).toBeDefined();
    if (!entry) throw new Error("Uploaded image has no library membership.");
    const trash = async (key: string) => {
      const preview = await request.get(`/api/artifact-library/${encodeURIComponent(entry.id)}/deletion-impact`);
      expect(preview.status()).toBe(200);
      const impact = await preview.json() as RecoveryImpact;
      const result = await request.post(`/api/artifact-library/${encodeURIComponent(entry.id)}/trash`, { headers, data: {
        expected_revision: impact.revision, impact_sha256: impact.impact_sha256, operation_key: key,
      } });
      expect(result.status()).toBe(200);
      return await result.json() as RecoveryItem;
    };
    const first = await trash(`trash-media-${width}`);
    await page.goto("/?view=settings&settings=data-and-backups");
    const center = page.locator("section", { has: page.getByRole("heading", { name: "Recently Deleted", exact: true }) });
    await center.getByRole("combobox", { name: /^Type/ }).selectOption("media_library_entry");
    const row = center.getByRole("article").filter({ hasText: title });
    await expect(row).toBeVisible();
    await expect(row.locator(`time[datetime="${first.purge_after}"]`)).toBeVisible();
    await expect(row.getByText("Recoverable · Media Library")).toBeVisible();
    await row.getByRole("button", { name: `Restore ${title}`, exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Restore this Media Library item?" });
    await expect(dialog.getByText(/favorites, collections and tags/)).toBeVisible();
    const restore = dialog.getByRole("button", { name: "Restore item", exact: true });
    await expect(restore).toBeEnabled();
    await restore.click();
    await expect(center.getByRole("status")).toHaveText("Media Library item restored with its favorites, collections and tags.");
    await expect(row).toHaveCount(0);
    const restored = await (await request.get("/api/artifact-library")).json() as { items: { id: string; artifact_id: string; favorite: boolean }[] };
    expect(restored.items.find((item) => item.artifact_id === artifact.id)).toMatchObject({ id: entry.id, favorite: true });
    await trash(`trash-media-again-${width}`);
    await expect(row).toBeVisible();
    await row.getByRole("button", { name: `Permanently delete ${title}`, exact: true }).click();
    const purgeDialog = page.getByRole("dialog", { name: "Permanently remove this Media Library item?" });
    await expect(purgeDialog.getByText(/No immediate storage reclamation is promised/)).toBeVisible();
    await purgeDialog.getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(row).toBeVisible();
    await row.getByRole("button", { name: `Permanently delete ${title}`, exact: true }).click();
    const purge = page.getByRole("dialog", { name: "Permanently remove this Media Library item?" }).getByRole("button", { name: "Delete permanently", exact: true });
    await expect(purge).toBeEnabled();
    await purge.click();
    await expect(center.getByRole("status")).toHaveText("Media Library item permanently removed. Shared and retained media remains available.");
    await expect(row).toHaveCount(0);
    const removed = await (await request.get("/api/artifact-library")).json() as { items: { id: string }[] };
    expect(removed.items.some((item) => item.id === entry.id)).toBe(false);
    expect((await request.get(`/api/artifacts/${encodeURIComponent(artifact.id)}/content`)).status()).toBe(200);
    expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1)).toBe(false);
  });
}
