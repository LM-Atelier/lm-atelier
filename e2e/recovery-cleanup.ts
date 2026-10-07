import { randomUUID } from "node:crypto";
import { expect, type APIRequestContext } from "@playwright/test";
import type { RecoveryImpact, RecoveryItem, RecoveryPage } from "../apps/web/src/recoveryTypes";

export async function permanentlyDeleteChat(request: APIRequestContext, chatId: string, headers: Record<string, string>) {
  const visible = await request.get(`/api/chats/${chatId}/metadata`);
  let item: RecoveryItem;
  if (visible.status() === 404) {
    let cursor: string | null = null;
    let deleted: RecoveryItem | undefined;
    do {
      const response = await request.get("/api/recovery-items", { params: {
        limit: 20, state: "recoverable", kind: "chat", ...(cursor ? { cursor } : {}),
      } });
      expect(response.status(), await response.text()).toBe(200);
      const page = await response.json() as RecoveryPage;
      deleted = page.items.find((candidate) => candidate.subject_id === chatId);
      cursor = page.next_cursor;
    } while (!deleted && cursor);
    expect(deleted).toBeDefined();
    if (!deleted) throw new Error("Chat has no recoverable deletion to clean up.");
    item = deleted;
  } else {
    expect(visible.status(), await visible.text()).toBe(200);
    const preview = await request.get(`/api/chats/${chatId}/deletion-impact`);
    expect(preview.status(), await preview.text()).toBe(200);
    const impact = await preview.json() as RecoveryImpact;
    const response = await request.post(`/api/chats/${chatId}/trash`, { headers, data: {
      expected_revision: impact.revision, impact_sha256: impact.impact_sha256, operation_key: randomUUID(),
    } });
    expect(response.status(), await response.text()).toBe(200);
    item = await response.json() as RecoveryItem;
  }
  const preview = await request.get(`/api/recovery-items/${item.deletion_id}/impact`);
  expect(preview.status(), await preview.text()).toBe(200);
  const impact = await preview.json() as RecoveryImpact;
  const purged = await request.post(`/api/recovery-items/${item.deletion_id}/purge`, { headers, data: {
    expected_revision: impact.revision, impact_sha256: impact.impact_sha256,
    operation_key: randomUUID(), acknowledgement: "permanently-delete",
  } });
  expect(purged.status(), await purged.text()).toBe(200);
  expect((await request.get(`/api/chats/${chatId}/metadata`)).status()).toBe(404);
}
