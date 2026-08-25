import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import {
  admitCreatedPromptBatch,
  admitQueuedPromptBatch,
  promptCountMaximum,
  type PromptDirectQueueAuthority,
} from "./promptDirectQueue";
import type { PromptBatch } from "./types";

const stamp = "2026-08-21T12:00:00Z";
const authority: PromptDirectQueueAuthority = {
  chatId: "chat-one",
  templateId: "template-one",
  revisionId: "revision-one",
  schemaVersion: 1,
  contractSha256: "a".repeat(64),
  itemCount: 2,
  selectionSeed: 42,
  queueIdempotencyKey: "queue-key",
};

function digest(prompt: string): string {
  return createHash("sha256")
    .update(`prompt-expansion-rendered-v1\0${prompt}`, "utf8")
    .digest("hex");
}

function draft(): PromptBatch {
  return {
    id: "batch-one",
    chat_id: authority.chatId,
    prompt_template_id: authority.templateId,
    prompt_template_revision_id: authority.revisionId,
    schema_version: authority.schemaVersion,
    contract_sha256: authority.contractSha256,
    codec_version: 2,
    requested_count: authority.itemCount,
    selection_seed: authority.selectionSeed,
    plan_sha256: "b".repeat(64),
    state: "draft",
    plan_version: 1,
    queue_idempotency_key: null,
    work_plan_id: null,
    queued_at: null,
    replayed: false,
    items: [1, 2].map((ordinal) => {
      const prompt = `Prompt ${ordinal}`;
      return {
        id: `item-${ordinal}`,
        ordinal,
        rendered_prompt: prompt,
        rendered_sha256: digest(prompt),
        reviewed_prompt: prompt,
        reviewed_sha256: digest(prompt),
        selected: true,
        review_version: 1,
        reroll_count: 0,
        work_step_id: null,
        run_id: null,
        media_seed: null,
      };
    }),
  };
}

function queued(queueKey = authority.queueIdempotencyKey): PromptBatch {
  const source = draft();
  return {
    ...source,
    state: "queued",
    plan_version: 2,
    queue_idempotency_key: queueKey,
    work_plan_id: "work-plan-one",
    queued_at: stamp,
    replayed: true,
    items: source.items.map((item) => ({
      ...item,
      work_step_id: `step-${item.ordinal}`,
      run_id: `run-${item.ordinal}`,
      media_seed: 100 + item.ordinal,
    })),
  };
}

describe("promptDirectQueue admission", () => {
  it("accepts an exact already-queued create replay without creating another batch", async () => {
    await expect(admitCreatedPromptBatch(queued(), authority))
      .resolves.toMatchObject({
        state: "queued",
        plan_version: 2,
        queue_idempotency_key: authority.queueIdempotencyKey,
      });
  });

  it("rejects a queued replay with another queue key or a partially linked item", async () => {
    await expect(admitCreatedPromptBatch(queued("other-key"), authority)).rejects.toThrow();
    const partial = queued();
    partial.items[1] = { ...partial.items[1], run_id: null };
    await expect(admitCreatedPromptBatch(partial, authority)).rejects.toThrow();
  });

  it("rejects edited draft items and tampered queue successors", async () => {
    const edited = draft();
    edited.items[0] = { ...edited.items[0], review_version: 2 };
    await expect(admitCreatedPromptBatch(edited, authority)).rejects.toThrow();

    const source = draft();
    const tampered = queued();
    tampered.items[0] = {
      ...tampered.items[0],
      rendered_prompt: "Different prompt",
      rendered_sha256: digest("Different prompt"),
      reviewed_prompt: "Different prompt",
      reviewed_sha256: digest("Different prompt"),
    };
    await expect(admitQueuedPromptBatch(tampered, authority, source)).rejects.toThrow();
  });

  it("bounds the casual prompt count by both runtime policy and protocol", () => {
    expect(promptCountMaximum(8)).toBe(8);
    expect(promptCountMaximum(99)).toBe(16);
    expect(promptCountMaximum(0)).toBe(1);
    expect(promptCountMaximum(Number.NaN)).toBe(1);
  });
});
