import { expect, test } from "@playwright/test";
import type { RecoveryImpact, RecoveryItem } from "../apps/web/src/recoveryTypes";

for (const [width, zoom] of [[1280, 1], [390, 1], [1280, 2]]) {
  test(`recovers Project settings without losing or refiling its chats at ${width}px and ${zoom * 100}% zoom`, async ({ page, request }) => {
    test.setTimeout(90_000);
    await page.setViewportSize({ width, height: 900 });
    if (zoom === 2) await page.addInitScript(() => {
      document.addEventListener("DOMContentLoaded", () => { document.documentElement.style.zoom = "2"; });
    });
    await page.route("**/api/setup/readiness", (route) => route.fulfill({ json: { version: 2, state: "ready", roles: [] } }));
    const { csrf_token } = await (await request.post("/api/session")).json() as { csrf_token: string };
    const headers = { "x-local-lm-csrf": csrf_token };
    const name = `Recoverable garden layout ${width}-${zoom}`;
    const createdProject = await request.post("/api/projects", { headers, data: {
      name, instructions: "Keep the garden paths wide", generation_settings_json: { chat: { max_tokens: 37 } },
    } });
    expect(createdProject.status()).toBe(201);
    const project = await createdProject.json() as { id: string };
    const other = await request.post("/api/projects", { headers, data: { name: `Garden paths ${width}-${zoom}` } });
    expect(other.status()).toBe(201);
    const destination = await other.json() as { id: string };
    const title = `Garden measurements ${width}-${zoom}`;
    const createdChat = await request.post("/api/chats", { headers, data: { title, project_id: project.id, routing_mode: "text" } });
    expect(createdChat.status()).toBe(201);
    const chat = await createdChat.json() as { id: string };
    const imported = await request.post("/api/models/import", { headers, data: {
      name: "Garden recovery chat fixture", role: "chat", engine: "mock", local_path: process.env.LM_ATELIER_E2E_MODEL_PATH,
    } });
    expect(imported.status()).toBe(201);
    const install = await imported.json() as { id: string };
    const profiles = await (await request.get("/api/profiles")).json() as { id: string; model_install_id: string | null }[];
    const profile = profiles.find((candidate) => candidate.model_install_id === install.id);
    if (!profile) throw new Error("Imported fixture has no chat profile.");
    expect((await request.patch(`/api/chats/${chat.id}`, { headers, data: { active_chat_profile_id: profile.id } })).status()).toBe(200);
    const accepted = await request.post(`/api/chats/${chat.id}/turns`, { headers, data: { text: "Keep the garden spacing", mode: "text", profile_id: profile.id } });
    expect(accepted.status()).toBe(202);
    const run = await accepted.json() as { run: { id: string } };
    await expect.poll(async () => (await (await request.get(`/api/runs/${run.run.id}`)).json()).status).toBe("complete");
    await page.addInitScript((id: string) => {
      localStorage.setItem("local-lm-chat", id);
      sessionStorage.setItem("lm-atelier-setup-dismissed", "1");
    }, chat.id);
    await page.goto("/");
    const composer = page.getByRole("textbox", { name: "Message", exact: true });
    await composer.fill("Unsent garden measurements");
    await expect.poll(async () => (await (await request.get(`/api/chats/${chat.id}/composer-draft`)).json()).text).toBe("Unsent garden measurements");
    const before = await (await request.get(`/api/chats/${chat.id}/messages`)).json() as { messages: { id: string }[] };
    expect(before.messages).toHaveLength(2);
    if (width < 700) await page.getByRole("button", { name: "Toggle navigation" }).click();
    await page.getByRole("button", { name: `Manage ${name}`, exact: true }).click();
    await page.getByRole("dialog", { name: "Manage project", exact: true }).getByRole("button", { name: "Delete project", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Move this project to Recently Deleted?" });
    const confirm = dialog.getByRole("button", { name: "Move to Recently Deleted", exact: true });
    await expect(confirm).toBeEnabled();
    await expect(dialog.getByText(/1 chats stay available/)).toBeVisible();
    expect((await request.patch(`/api/projects/${project.id}`, { headers, data: { description: "Wide paths" } })).status()).toBe(200);
    const refusal = page.waitForResponse((response) => response.url().endsWith(`/api/projects/${project.id}/trash`) && response.request().method() === "POST");
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
    await expect(undo).toBeFocused();
    const unfiled = await request.get(`/api/chats/${chat.id}/metadata`);
    expect(unfiled.status()).toBe(200);
    expect((await unfiled.json()).project_id).toBeNull();
    expect((await request.get(`/api/projects/${project.id}`)).status()).toBe(404);
    expect(await (await request.get(`/api/chats/${chat.id}/messages`)).json()).toEqual(before);
    expect((await request.patch(`/api/chats/${chat.id}`, { headers, data: { project_id: destination.id } })).status()).toBe(200);
    await undo.press("Enter");
    await expect(undo).toHaveCount(0);
    expect((await (await request.get(`/api/chats/${chat.id}/metadata`)).json()).project_id).toBe(destination.id);
    if (width < 700) await page.getByRole("button", { name: "Toggle navigation" }).click();
    await expect(composer).toHaveValue("Unsent garden measurements");
    const trash = async (key: string) => {
      const preview = await request.get(`/api/projects/${project.id}/deletion-impact`);
      expect(preview.status()).toBe(200);
      const impact = await preview.json() as RecoveryImpact;
      const moved = await request.post(`/api/projects/${project.id}/trash`, { headers, data: {
        expected_revision: impact.revision, impact_sha256: impact.impact_sha256, operation_key: key,
      } });
      expect(moved.status()).toBe(200);
      return await moved.json() as RecoveryItem;
    };
    const deleted = await trash(`trash-settings-project-${width}-${zoom}`);
    await page.goto("/?view=settings&settings=data-and-backups");
    const center = page.locator("section", { has: page.getByRole("heading", { name: "Recently Deleted", exact: true }) });
    await center.getByRole("combobox", { name: /^Type/ }).selectOption("project");
    const row = center.getByRole("article").filter({ hasText: name });
    await expect(row).toBeVisible();
    await expect(row.locator(`time[datetime="${deleted.purge_after}"]`)).toBeVisible();
    await row.getByRole("button", { name: `Restore ${name}`, exact: true }).click();
    const restoreDialog = page.getByRole("dialog", { name: "Restore this project?" });
    await expect(restoreDialog.getByText(/Chats moved elsewhere stay there/)).toBeVisible();
    await expect(restoreDialog.getByRole("button", { name: "Restore project", exact: true })).toBeEnabled();
    await restoreDialog.getByRole("button", { name: "Restore project", exact: true }).click();
    await expect(row).toHaveCount(0);
    expect(await (await request.get(`/api/projects/${project.id}`)).json()).toMatchObject({ id: project.id,
      description: "Wide paths", instructions: "Keep the garden paths wide", generation_settings_json: { chat: { max_tokens: 37 } } });
    expect((await (await request.get(`/api/chats/${chat.id}/metadata`)).json()).project_id).toBe(destination.id);
    await trash(`trash-project-purge-${width}-${zoom}`);
    await expect(row).toBeVisible();
    await row.getByRole("button", { name: `Permanently delete ${name}`, exact: true }).click();
    const purgeDialog = page.getByRole("dialog", { name: "Permanently delete this project?" });
    await expect(purgeDialog.getByText(/Its chats and media remain available/)).toBeVisible();
    await purgeDialog.getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(row).toBeVisible();
    await row.getByRole("button", { name: `Permanently delete ${name}`, exact: true }).click();
    const purge = page.getByRole("dialog", { name: "Permanently delete this project?" }).getByRole("button", { name: "Delete permanently", exact: true });
    await expect(purge).toBeEnabled();
    await purge.click();
    await expect(row).toHaveCount(0);
    await expect(center.getByRole("status")).toHaveText("Project permanently deleted. Its chats and media remain available.");
    expect(await (await request.get(`/api/chats/${chat.id}/messages`)).json()).toEqual(before);
    expect((await (await request.get(`/api/chats/${chat.id}/metadata`)).json()).project_id).toBe(destination.id);
    expect((await (await request.get(`/api/chats/${chat.id}/composer-draft`)).json()).text).toBe("Unsent garden measurements");
    expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1)).toBe(false);
  });
}
