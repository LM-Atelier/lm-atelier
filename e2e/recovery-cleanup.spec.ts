import { expect, test } from "@playwright/test";
import type { RecoveryImpact } from "../apps/web/src/recoveryTypes";
import { permanentlyDeleteChat } from "./recovery-cleanup";

test("permanently cleans up a chat already moved to recovery", async ({ request }) => {
  const session = await request.post("/api/session");
  expect(session.status()).toBe(200);
  const { csrf_token } = await session.json() as { csrf_token: string };
  const headers = { "x-local-lm-csrf": csrf_token };
  const created = await request.post("/api/chats", { headers, data: { title: "Garden cleanup notes" } });
  expect(created.status()).toBe(201);
  const chat = await created.json() as { id: string };
  const preview = await request.get(`/api/chats/${chat.id}/deletion-impact`);
  expect(preview.status()).toBe(200);
  const impact = await preview.json() as RecoveryImpact;
  const trash = await request.post(`/api/chats/${chat.id}/trash`, { headers, data: {
    expected_revision: impact.revision, impact_sha256: impact.impact_sha256, operation_key: `trash-cleanup-${chat.id}`,
  } });
  expect(trash.status()).toBe(200);
  await permanentlyDeleteChat(request, chat.id, headers);
  expect((await request.get(`/api/chats/${chat.id}/metadata`)).status()).toBe(404);
});
