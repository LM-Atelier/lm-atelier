import type { PromptBatch, PromptBatchItem, RoutingMode } from "./types";

export interface ComposerPromptSource {
  version: 1;
  batch_id: string;
  expected_plan_version: number;
  expected_plan_sha256: string;
  item_id: string;
  expected_review_version: number;
  expected_reviewed_sha256: string;
  prompt_template_id: string;
  prompt_template_revision_id: string;
  contract_sha256: string;
}

export interface ComposerDraft {
  text: string;
  promptSource: ComposerPromptSource | null;
}

export type ComposerDraftUpdate = ComposerDraft | ((current: ComposerDraft) => ComposerDraft);

export interface PromptComposerInsertion {
  text: string;
  source: ComposerPromptSource;
}

export const EMPTY_COMPOSER_DRAFT: ComposerDraft = {
  text: "",
  promptSource: null,
};

export function insertionForPromptBatchItem(
  batch: PromptBatch,
  item: PromptBatchItem,
): PromptComposerInsertion {
  return {
    text: item.reviewed_prompt,
    source: {
      version: 1,
      batch_id: batch.id,
      expected_plan_version: batch.plan_version,
      expected_plan_sha256: batch.plan_sha256,
      item_id: item.id,
      expected_review_version: item.review_version,
      expected_reviewed_sha256: item.reviewed_sha256,
      prompt_template_id: batch.prompt_template_id,
      prompt_template_revision_id: batch.prompt_template_revision_id,
      contract_sha256: batch.contract_sha256,
    },
  };
}

export function insertedComposerDraft(insertion: PromptComposerInsertion): ComposerDraft {
  return { text: insertion.text, promptSource: insertion.source };
}

export function composerDraftWithText(
  draft: ComposerDraft,
  text: string,
): ComposerDraft {
  return {
    text,
    promptSource: text.trim() ? draft.promptSource : null,
  };
}

export function detachedComposerDraft(draft: ComposerDraft): ComposerDraft {
  return draft.promptSource ? { ...draft, promptSource: null } : draft;
}

export function updatedComposerDrafts(
  drafts: Record<string, ComposerDraft>,
  chatId: string,
  update: ComposerDraftUpdate,
): Record<string, ComposerDraft> {
  const previous = drafts[chatId] ?? EMPTY_COMPOSER_DRAFT;
  const next = typeof update === "function" ? update(previous) : update;
  return { ...drafts, [chatId]: next };
}

export function withoutComposerDraft(
  drafts: Record<string, ComposerDraft>,
  chatId: string,
): Record<string, ComposerDraft> {
  if (!(chatId in drafts)) return drafts;
  const next = { ...drafts };
  delete next[chatId];
  return next;
}

export function promptSourceForTurn(
  draft: ComposerDraft,
  mode: RoutingMode,
  inputArtifactCount: number,
  referenceCount: number,
  outputCount: number | undefined,
): ComposerPromptSource | undefined {
  return mode === "image"
    && inputArtifactCount === 0
    && referenceCount === 0
    && outputCount === undefined
    ? draft.promptSource ?? undefined
    : undefined;
}
