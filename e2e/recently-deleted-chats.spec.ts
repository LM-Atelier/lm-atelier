import { permanentlyDeleteChat } from "./recovery-cleanup";
import { expect, test } from "@playwright/test";
import type { RecoveryImpact, RecoveryItem } from "../apps/web/src/recoveryTypes";

for (const width of [1280, 390]) {
  test(`moves a chat to recovery and undoes with its transcript and draft at ${width}px`, async ({ page, request }) => {
    test.setTimeout(90_000);
    await page.setViewportSize({ width, height: 900 });
    await page.route("**/api/setup/readiness", (route) => route.fulfill({ json: { version: 2, state: "ready", roles: [] } }));
    const session = await request.post("/api/session");
    const { csrf_token: csrf } = await session.json() as { csrf_token: string };
    const headers = { "x-local-lm-csrf": csrf };
    const imported = await request.post("/api/models/import", { headers, data: {
      name: "Garden recovery fixture", role: "chat", engine: "mock", local_path: process.env.LM_ATELIER_E2E_MODEL_PATH,
    } });
    expect(imported.status()).toBe(201);
    const install = await imported.json() as { id: string };
    const profilesResponse = await request.get("/api/profiles");
    expect(profilesResponse.status()).toBe(200);
    const profiles = await profilesResponse.json() as { id: string; model_install_id: string | null }[];
    const profile = profiles.find((item) => item.model_install_id === install.id);
    expect(profile).toBeDefined();
    if (!profile) throw new Error("Imported fixture has no chat profile.");
    const title = `Recoverable garden conversation ${width}`;
    const created = await request.post("/api/chats", { headers, data: { title, routing_mode: "text" } });
    expect(created.status()).toBe(201);
    const chat = await created.json() as { id: string };
    try {
      const selected = await request.patch(`/api/chats/${chat.id}`, { headers, data: { active_chat_profile_id: profile.id } });
      expect(selected.status()).toBe(200);
      const accepted = await request.post(`/api/chats/${chat.id}/turns`, { headers, data: { text: "Keep the garden spacing", mode: "text", profile_id: profile.id } });
      expect(accepted.status()).toBe(202);
      const run = await accepted.json() as { run: { id: string } };
      await expect.poll(async () => (await (await request.get(`/api/runs/${run.run.id}`)).json()).status).toBe("complete");
      const before = await (await request.get(`/api/chats/${chat.id}/messages`)).json() as { messages: { id: string }[] };
      expect(before.messages).toHaveLength(2);
      await page.addInitScript((id: string) => { localStorage.setItem("local-lm-chat", id); }, chat.id);
      await page.goto("/");
      const composer = page.getByRole("textbox", { name: "Message", exact: true });
      await composer.fill("Unsent garden measurements");
      await expect.poll(async () => (await (await request.get(`/api/chats/${chat.id}/composer-draft`)).json()).text).toBe("Unsent garden measurements");
      if (width < 700) await page.getByRole("button", { name: "Toggle navigation" }).click();
      await page.getByRole("button", { name: `Manage ${title}`, exact: true }).click();
      await page.getByRole("checkbox", { name: "Delete generated media with chat" }).check();
      await page.getByRole("button", { name: "Delete chat", exact: true }).click();
      const confirm = page.getByRole("dialog", { name: "Move this chat to Recently Deleted?" });
      await expect(confirm.getByText(/2 messages · 0 media files/)).toBeVisible();
      const move = confirm.getByRole("button", { name: "Move to Recently Deleted", exact: true });
      await expect(move).toBeEnabled();
      const draft = await (await request.get(`/api/chats/${chat.id}/composer-draft`)).json() as { revision: number };
      const changed = await request.put(`/api/chats/${chat.id}/composer-draft`, { headers, data: {
        expected_revision: draft.revision, draft: { text: "Unsent garden measurements", mode: "text" },
      } });
      expect(changed.status()).toBe(200);
      const refusal = page.waitForResponse((response) => response.url().endsWith(`/api/chats/${chat.id}/trash`) && response.request().method() === "POST");
      await move.click();
      expect((await refusal).status()).toBe(409);
      await expect(confirm).toBeVisible();
      await expect(move).toBeDisabled();
      await confirm.getByRole("button", { name: "Check again", exact: true }).click();
      await expect(move).toBeEnabled();
      await move.click();
      const undo = page.getByRole("button", { name: "Undo", exact: true });
      await expect(undo).toBeVisible();
      expect((await request.get(`/api/chats/${chat.id}`)).status()).toBe(404);
      const deleted = await (await request.get("/api/recovery-items")).json() as { items: RecoveryItem[] };
      expect(deleted.items.find((item) => item.subject_id === chat.id)?.delete_generated_media).toBe(true);
      const restoration = page.waitForResponse((response) => response.url().endsWith("/restore") && response.request().method() === "POST");
      await undo.click();
      expect((await restoration).status()).toBe(200);
      await expect(page.locator(".recovery-notice")).toHaveCount(0);
      expect((await request.get(`/api/chats/${chat.id}`)).status()).toBe(200);
      const restored = await (await request.get(`/api/chats/${chat.id}/messages`)).json() as { messages: { id: string }[] };
      expect(restored.messages.map((message) => message.id)).toEqual(before.messages.map((message) => message.id));
      if (width < 700) await page.getByRole("button", { name: "Toggle navigation" }).click();
      await expect(composer).toHaveValue("Unsent garden measurements");
    } finally {
      await page.goto("about:blank");
      await permanentlyDeleteChat(request, chat.id, headers);
    }
  });

  test(`restores and permanently deletes a chat from Settings at ${width}px`, async ({ page, request }) => {
    test.setTimeout(90_000);
    await page.setViewportSize({ width, height: 900 });
    await page.route("**/api/setup/readiness", (route) => route.fulfill({ json: { version: 2, state: "ready", roles: [] } }));
    const session = await request.post("/api/session");
    const { csrf_token: csrf } = await session.json() as { csrf_token: string };
    const headers = { "x-local-lm-csrf": csrf };
    const title = `Deleted garden notes ${width}`;
    const created = await request.post("/api/chats", { headers, data: { title } });
    expect(created.status()).toBe(201);
    const chat = await created.json() as { id: string };
    const trash = async (key: string) => {
      const preview = await request.get(`/api/chats/${chat.id}/deletion-impact`);
      expect(preview.status()).toBe(200);
      const impact = await preview.json() as RecoveryImpact;
      const response = await request.post(`/api/chats/${chat.id}/trash`, { headers, data: {
        expected_revision: impact.revision, impact_sha256: impact.impact_sha256, operation_key: key,
      } });
      expect(response.status()).toBe(200);
      return await response.json() as RecoveryItem;
    };
    const first = await trash(`trash-garden-${width}`);
    await page.goto("/?view=settings&settings=data-and-backups");
    const center = page.locator("section", { has: page.getByRole("heading", { name: "Recently Deleted", exact: true }) });
    const row = center.getByRole("article").filter({ hasText: title });
    await expect(row).toBeVisible();
    await expect(row.locator(`time[datetime="${first.purge_after}"]`)).toBeVisible();
    await row.getByRole("button", { name: `Restore ${title}`, exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Restore this chat?" });
    await expect(dialog).toBeVisible();
    const restore = dialog.getByRole("button", { name: "Restore chat", exact: true });
    await expect(restore).toBeEnabled();
    await restore.click();
    await expect(center.getByRole("status")).toHaveText("Chat restored with its original history.");
    await expect(row).toHaveCount(0);
    const restored = await request.get(`/api/chats/${chat.id}`);
    expect(restored.status()).toBe(200);
    expect((await restored.json()).id).toBe(chat.id);

    await trash(`trash-garden-again-${width}`);
    await expect(row).toBeVisible();
    await row.getByRole("button", { name: `Permanently delete ${title}`, exact: true }).click();
    const purgeDialog = page.getByRole("dialog", { name: "Permanently delete this chat?" });
    await expect(purgeDialog).toBeVisible();
    await expect(purgeDialog.getByText(/This cannot be undone/)).toBeVisible();
    await purgeDialog.getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(row).toBeVisible();
    await row.getByRole("button", { name: `Permanently delete ${title}`, exact: true }).click();
    const purge = page.getByRole("dialog", { name: "Permanently delete this chat?" }).getByRole("button", { name: "Delete permanently", exact: true });
    await expect(purge).toBeEnabled();
    await purge.click();
    await expect(center.getByRole("status")).toHaveText("Chat permanently deleted. Shared and retained media remains available.");
    await expect(row).toHaveCount(0);
    expect((await request.get(`/api/chats/${chat.id}`)).status()).toBe(404);
    expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1)).toBe(false);
  });
}
