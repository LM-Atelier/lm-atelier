import { expect, type APIRequestContext } from "@playwright/test";
import type { RecoveryImpact, RecoveryItem } from "../apps/web/src/recoveryTypes";

export async function permanentlyDeleteFixture(
  request: APIRequestContext,
  headers: Record<string, string>,
  kind: "chat" | "project",
  subjectId: string,
  deletionId?: string,
): Promise<void> {
  const collection = kind === "chat" ? "chats" : "projects";
  if (!deletionId) {
    const preview = await request.get(`/api/${collection}/${subjectId}/deletion-impact`);
    expect(preview.status()).toBe(200);
    const impact = await preview.json() as RecoveryImpact;
    const trashed = await request.post(`/api/${collection}/${subjectId}/trash`, { headers, data: {
      expected_revision: impact.revision, impact_sha256: impact.impact_sha256,
      operation_key: `trash-fixture-${subjectId}`, ...(kind === "chat" ? { delete_generated_media: false } : {}),
    } });
    expect(trashed.status()).toBe(200);
    deletionId = (await trashed.json() as RecoveryItem).deletion_id;
  }
  const preview = await request.get(`/api/recovery-items/${deletionId}/impact`);
  expect(preview.status()).toBe(200);
  const impact = await preview.json() as RecoveryImpact;
  const purged = await request.post(`/api/recovery-items/${deletionId}/purge`, { headers, data: {
    expected_revision: impact.revision, impact_sha256: impact.impact_sha256,
    operation_key: `purge-fixture-${subjectId}`, acknowledgement: "permanently-delete",
  } });
  expect(purged.status()).toBe(200);
  expect((await purged.json() as { reclaimed_bytes: number }).reclaimed_bytes).toBe(0);
}
