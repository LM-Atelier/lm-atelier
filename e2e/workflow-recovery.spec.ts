import { expect, test } from "@playwright/test";
import type { Workflow, WorkflowFamily } from "../apps/web/src/types";
import type { RecoveryImpact, RecoveryItem } from "../apps/web/src/recoveryTypes";

for (const [width, zoom] of [[1280, 1], [390, 1], [1280, 2]]) {
  test(`recovers a workflow without restoring execution authority at ${width}px and ${zoom * 100}% zoom`, async ({ page, request }) => {
    test.setTimeout(90_000);
    await page.setViewportSize({ width, height: 900 });
    if (zoom === 2) await page.addInitScript(() => {
      document.addEventListener("DOMContentLoaded", () => { document.documentElement.style.zoom = "2"; });
    });
    await page.route("**/api/setup/readiness", (route) => route.fulfill({ json: { version: 2, state: "ready", roles: [] } }));
    await page.addInitScript(() => { sessionStorage.setItem("lm-atelier-setup-dismissed", "1"); });
    const { csrf_token } = await (await request.post("/api/session")).json() as { csrf_token: string };
    const headers = { "x-local-lm-csrf": csrf_token };
    const name = `Recoverable garden workflow ${width}-${zoom}`;
    const created = await request.post("/api/workflows", { headers, data: {
      name, operation: "text_to_image", engine: "mock", api_graph: { garden: { class_type: "NeutralFixture", inputs: { spacing: 37 } } },
    } });
    expect(created.status()).toBe(201);
    const workflow = await created.json() as Workflow;
    const original = await (await request.get(`/api/workflows/${workflow.id}`)).json() as Workflow;
    expect(workflow.family_id).toBeTruthy();
    const familyPath = `/api/workflow-families/${workflow.family_id}`;
    expect((await request.put(`${familyPath}/preferences/image`, { headers, data: { enabled: true, is_default: true, sort_order: 0 } })).status()).toBe(200);
    await page.goto("/?view=workflows");
    await page.getByRole("searchbox", { name: "Search workflow families" }).fill(name);
    await page.getByRole("button", { name: new RegExp(name) }).first().click();
    const remove = page.getByRole("button", { name: "Delete workflow family", exact: true });
    await remove.click();
    const dialog = page.getByRole("dialog", { name: "Move this workflow family to Recently Deleted?" });
    const confirm = dialog.getByRole("button", { name: "Move to Recently Deleted", exact: true });
    await expect(dialog.getByRole("alert")).toHaveText(/selected by a chat or project, or is set as a default/);
    await expect(confirm).toBeDisabled();
    await dialog.getByRole("button", { name: "Cancel", exact: true }).click();
    expect((await (await request.get(familyPath)).json() as WorkflowFamily).preferences.some((preference) => preference.is_default)).toBe(true);
    expect((await request.put(`${familyPath}/preferences/image`, { headers, data: { enabled: true, is_default: false, sort_order: 0 } })).status()).toBe(200);
    await remove.click();
    await expect(confirm).toBeEnabled();
    expect((await request.patch(familyPath, { headers, data: { description: "Wide garden paths" } })).status()).toBe(200);
    const refusal = page.waitForResponse((response) => response.url().endsWith(`${familyPath}/trash`) && response.request().method() === "POST");
    await confirm.click(); expect((await refusal).status()).toBe(409);
    await expect(dialog).toBeVisible(); await expect(confirm).toBeDisabled();
    await dialog.getByRole("button", { name: "Check again", exact: true }).click();
    await expect(confirm).toBeEnabled();
    const moved = page.waitForResponse((response) => response.url().endsWith(`${familyPath}/trash`) && response.request().method() === "POST" && response.status() === 200);
    await confirm.focus(); await confirm.press("Enter");
    const item = await (await moved).json() as RecoveryItem;
    await expect(dialog).toHaveCount(0);
    const undo = page.getByRole("button", { name: "Undo", exact: true });
    await expect(undo).toBeFocused();
    await expect(page.locator(`time[datetime="${item.purge_after}"]`)).toBeVisible();
    await expect(page.getByRole("button", { name: new RegExp(name) })).toHaveCount(0);
    expect((await request.get(`/api/workflows/${workflow.id}`)).status()).toBe(404);
    await undo.press("Enter"); await expect(undo).toHaveCount(0);
    await expect(page.getByRole("heading", { name, exact: true }).last()).toBeVisible();
    const restored = await (await request.get(familyPath)).json() as WorkflowFamily;
    expect(restored).toMatchObject({ id: workflow.family_id, enabled: false, archived: false, description: "Wide garden paths" });
    expect(restored.preferences.every((preference) => !preference.enabled && !preference.is_default)).toBe(true);
    const detail = await (await request.get(`/api/workflows/${workflow.id}`)).json() as Workflow;
    expect(detail.current_revision_id).toBe(workflow.current_revision_id);
    expect(detail.revisions).toEqual(original.revisions);
    expect(detail.revisions.every((revision) => !revision.trusted)).toBe(true);
    const preview = await (await request.get(`${familyPath}/deletion-impact`)).json() as RecoveryImpact;
    const deleted = await request.post(`${familyPath}/trash`, { headers, data: {
      expected_revision: preview.revision, impact_sha256: preview.impact_sha256, operation_key: `trash-center-workflow-${width}-${zoom}`,
    } });
    expect(deleted.status()).toBe(200);
    await page.goto("/?view=settings&settings=data-and-backups");
    const center = page.locator("section", { has: page.getByRole("heading", { name: "Recently Deleted", exact: true }) });
    await center.getByRole("combobox", { name: /^Type/ }).selectOption("workflow_family");
    const row = center.getByRole("article").filter({ hasText: name });
    await row.getByRole("button", { name: `Restore ${name}`, exact: true }).click();
    const restoreDialog = page.getByRole("dialog", { name: "Restore this workflow?" });
    await expect(restoreDialog.getByText(/disabled/)).toBeVisible();
    const restore = restoreDialog.getByRole("button", { name: "Restore workflow", exact: true });
    await expect(restore).toBeEnabled(); await restore.click();
    await expect(row).toHaveCount(0);
    expect(await (await request.get(familyPath)).json()).toMatchObject({ id: workflow.family_id, enabled: false });
    expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1)).toBe(false);
  });
}
