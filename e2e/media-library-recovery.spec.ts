import { expect, test } from "@playwright/test";
import type { RecoveryImpact } from "../apps/web/src/recoveryTypes";

for (const [width, zoom] of [[1280, 1], [390, 1], [1280, 2]]) {
  test(`moves a library item with a fresh confirmation and Undo at ${width}px and ${zoom * 100}% zoom`, async ({ page, request }) => {
    test.setTimeout(90_000);
    await page.setViewportSize({ width, height: 900 });
    if (zoom === 2) await page.addInitScript(() => {
      document.addEventListener("DOMContentLoaded", () => { document.documentElement.style.zoom = "2"; });
    });
    await page.route("**/api/setup/readiness", (route) => route.fulfill({ json: { version: 2, state: "ready", roles: [] } }));
    const session = await request.post("/api/session");
    const { csrf_token: csrf } = await session.json() as { csrf_token: string };
    const headers = { "x-local-lm-csrf": csrf };
    const name = "Garden recovery.png";
    const uploaded = await request.post("/api/artifacts?kind=image", { headers, multipart: {
      file: { name, mimeType: "image/png", buffer: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a6V8AAAAASUVORK5CYII=", "base64") },
    } });
    expect(uploaded.status()).toBe(201);
    const artifact = await uploaded.json() as { id: string };
    const artifactPath = `/api/artifacts/${encodeURIComponent(artifact.id)}`;
    expect((await request.patch(artifactPath, { headers, data: { favorite: true } })).status()).toBe(200);
    const before = await (await request.get("/api/artifact-library")).json() as { items: { id: string; artifact_id: string }[] };
    const entry = before.items.find((item) => item.artifact_id === artifact.id);
    expect(entry).toBeDefined();
    if (!entry) throw new Error("Uploaded fixture has no library membership.");
    await page.goto("/?view=media");
    await page.getByRole("textbox", { name: "Search media" }).fill("Garden");
    const move = page.getByRole("button", { name: `Move ${name} to Recently Deleted`, exact: true });
    await expect(move).toBeVisible();
    await move.focus();
    await move.press("Enter");
    const dialog = page.getByRole("dialog", { name: "Move this Media Library item to Recently Deleted?" });
    const confirm = dialog.getByRole("button", { name: "Move to Recently Deleted", exact: true });
    await expect(confirm).toBeEnabled();
    expect((await request.patch(artifactPath, { headers, data: { favorite: false } })).status()).toBe(200);
    const refusal = page.waitForResponse((response) => response.url().endsWith(`/api/artifact-library/${encodeURIComponent(entry.id)}/trash`) && response.request().method() === "POST");
    await confirm.click();
    expect((await refusal).status()).toBe(409);
    await expect(dialog).toBeVisible();
    await expect(confirm).toBeDisabled();
    await dialog.getByRole("button", { name: "Check again", exact: true }).click();
    await expect(confirm).toBeEnabled();
    await confirm.focus();
    await confirm.press("Enter");
    await expect(dialog).toHaveCount(0);
    const undo = page.getByRole("button", { name: "Undo", exact: true });
    await expect(undo).toBeVisible();
    await expect(undo).toBeFocused();
    await expect(move).toHaveCount(0);
    expect((await request.get(`${artifactPath}/content`)).status()).toBe(200);
    // The button is renamed while the restore runs, so its going away is not the restore finishing.
    const restoration = page.waitForResponse((response) => response.url().endsWith("/restore") && response.request().method() === "POST");
    await undo.press("Enter");
    expect((await restoration).status()).toBe(200);
    await expect(undo).toHaveCount(0);
    await expect(move).toBeVisible();
    await expect(page.getByRole("textbox", { name: "Search media" })).toHaveValue("Garden");
    const restored = await (await request.get("/api/artifact-library")).json() as { items: { id: string; artifact_id: string; favorite: boolean }[] };
    expect(restored.items.find((item) => item.artifact_id === artifact.id)).toMatchObject({ id: entry.id, favorite: false });
    expect((await request.get(`${artifactPath}/content`)).status()).toBe(200);
    expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1)).toBe(false);
  });
}

test("reimporting the same file restores its library card through the live update", async ({ page, request }) => {
  await page.route("**/api/setup/readiness", (route) => route.fulfill({ json: { version: 2, state: "ready", roles: [] } }));
  const { csrf_token } = await (await request.post("/api/session")).json() as { csrf_token: string };
  const headers = { "x-local-lm-csrf": csrf_token };
  const name = "Garden recovery.png";
  const multipart = { file: { name, mimeType: "image/png", buffer: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a6V8AAAAASUVORK5CYII=", "base64") } };
  const uploaded = await request.post("/api/artifacts?kind=image", { headers, multipart });
  expect(uploaded.status()).toBe(201);
  const artifact = await uploaded.json() as { id: string };
  const artifactPath = `/api/artifacts/${encodeURIComponent(artifact.id)}`;
  expect((await request.patch(artifactPath, { headers, data: { favorite: true } })).status()).toBe(200);
  const library = await (await request.get("/api/artifact-library")).json() as { items: { id: string; artifact_id: string }[] };
  const entry = library.items.find((item) => item.artifact_id === artifact.id);
  if (!entry) throw new Error("Uploaded fixture has no library membership.");
  await page.goto("/?view=media");
  const search = page.getByRole("textbox", { name: "Search media" });
  await search.fill("Garden");
  const card = page.getByRole("button", { name: `Move ${name} to Recently Deleted`, exact: true });
  await expect(card).toBeVisible();
  const preview = await (await request.get(`/api/artifact-library/${encodeURIComponent(entry.id)}/deletion-impact`)).json() as RecoveryImpact;
  const removed = await request.post(`/api/artifact-library/${encodeURIComponent(entry.id)}/trash`, { headers, data: {
    expected_revision: preview.revision, impact_sha256: preview.impact_sha256, operation_key: "trash-for-live-reimport",
  } });
  expect(removed.status()).toBe(200);
  await expect(card).toHaveCount(0);
  const imported = await request.post("/api/artifacts", { headers, multipart });
  expect(imported.status()).toBe(201);
  expect((await imported.json()).id).toBe(artifact.id);
  await expect(card).toBeVisible();
  await expect(search).toHaveValue("Garden");
  const restored = await (await request.get("/api/artifact-library")).json() as { items: { id: string; artifact_id: string; favorite: boolean }[] };
  expect(restored.items.filter((item) => item.artifact_id === artifact.id)).toEqual([
    expect.objectContaining({ id: entry.id, artifact_id: artifact.id, favorite: true }),
  ]);
  expect((await (await request.get("/api/recovery-items")).json()).items).toEqual([]);
  expect((await request.get(`${artifactPath}/content`)).status()).toBe(200);
});
