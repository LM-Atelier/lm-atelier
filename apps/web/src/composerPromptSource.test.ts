import { describe, expect, it } from "vitest";
import {
  EMPTY_COMPOSER_DRAFT,
  composerDraftWithText,
  detachedComposerDraft,
  insertedComposerDraft,
  insertionForPromptBatchItem,
  promptSourceForTurn,
} from "./composerPromptSource";
import type { PromptBatch } from "./types";

const batch: PromptBatch = {
  id: "prompt-batch-one",
  chat_id: "chat-one",
  prompt_template_id: "ptdef-one",
  prompt_template_revision_id: "ptrev-two",
  schema_version: 1,
  contract_sha256: "a".repeat(64),
  codec_version: 2,
  requested_count: 1,
  selection_seed: 7,
  plan_sha256: "b".repeat(64),
  state: "draft",
  plan_version: 3,
  replayed: false,
  items: [{
    id: "prompt-item-one",
    ordinal: 1,
    rendered_prompt: "A portrait of Ada.",
    rendered_sha256: "c".repeat(64),
    reviewed_prompt: "A reviewed portrait of Ada.",
    reviewed_sha256: "d".repeat(64),
    selected: true,
    review_version: 2,
    reroll_count: 0,
  }],
};

describe("composer Prompt Library provenance", () => {
  it("inserts one reviewed item with the exact optimistic authority fields", () => {
    const insertion = insertionForPromptBatchItem(batch, batch.items[0]);

    expect(insertedComposerDraft(insertion)).toEqual({
      text: "A reviewed portrait of Ada.",
      promptSource: {
        version: 1,
        batch_id: "prompt-batch-one",
        expected_plan_version: 3,
        expected_plan_sha256: "b".repeat(64),
        item_id: "prompt-item-one",
        expected_review_version: 2,
        expected_reviewed_sha256: "d".repeat(64),
        prompt_template_id: "ptdef-one",
        prompt_template_revision_id: "ptrev-two",
        contract_sha256: "a".repeat(64),
      },
    });
  });

  it("retains source through edits, but detaches it when text is cleared", () => {
    const inserted = insertedComposerDraft(insertionForPromptBatchItem(batch, batch.items[0]));

    const edited = composerDraftWithText(inserted, "A warmer reviewed portrait of Ada.");
    expect(edited.promptSource).toBe(inserted.promptSource);
    expect(composerDraftWithText(edited, "   ")).toEqual({ text: "   ", promptSource: null });
  });

  it("detaches conservatively without changing text and keeps chats isolated", () => {
    const inserted = insertedComposerDraft(insertionForPromptBatchItem(batch, batch.items[0]));
    const drafts = {
      "chat-one": inserted,
      "chat-two": EMPTY_COMPOSER_DRAFT,
    };

    // Chip removal, adding an attachment, and choosing a reference all use
    // this same narrowing operation: prose remains, authority does not.
    expect(detachedComposerDraft(drafts["chat-one"])).toEqual({
      text: inserted.text,
      promptSource: null,
    });
    expect(drafts["chat-two"]).toBe(EMPTY_COMPOSER_DRAFT);
  });

  it("refuses to send source authority with attachments, references, or a non-image mode", () => {
    const inserted = insertedComposerDraft(insertionForPromptBatchItem(batch, batch.items[0]));

    expect(promptSourceForTurn(inserted, "image", 0, 0)).toBe(inserted.promptSource);
    expect(promptSourceForTurn(inserted, "image", 1, 0)).toBeUndefined();
    expect(promptSourceForTurn(inserted, "image", 0, 1)).toBeUndefined();
    expect(promptSourceForTurn(inserted, "video", 0, 0)).toBeUndefined();
    expect(promptSourceForTurn(inserted, "auto", 0, 0)).toBeUndefined();
  });
});
