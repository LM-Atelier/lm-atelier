import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, RefreshCw } from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import { api } from "./api";
import {
  insertionForPromptBatchItem,
  type PromptComposerInsertion,
} from "./composerPromptSource";
import { ErrorCallout } from "./ErrorCallout";
import type {
  PromptBatch,
  PromptBatchCreateInput,
  PromptBatchItem,
  PromptBatchItemUpdateInput,
  PromptBatchQueueInput,
  PromptTemplateDetail,
  PromptTemplateSlot,
} from "./types";

const MIN_ITEMS = 1;
const MAX_ITEMS = 16;
const MAX_SELECTION_SEED = 2_147_483_647;
const MAX_INPUT_CHARACTERS = 2_000;
const MAX_REVIEW_CHARACTERS = 32_000;
const MAX_IDENTIFIER_CHARACTERS = 40;
const SHA256 = /^[0-9a-f]{64}$/;
const BATCH_KEYS = [
  "id", "chat_id", "prompt_template_id", "prompt_template_revision_id",
  "schema_version", "contract_sha256", "codec_version", "requested_count",
  "selection_seed", "plan_sha256", "state", "plan_version", "items", "replayed",
  "queue_idempotency_key", "work_plan_id", "queued_at",
] as const;
const ITEM_KEYS = [
  "id", "ordinal", "rendered_prompt", "rendered_sha256", "reviewed_prompt",
  "reviewed_sha256", "selected", "review_version", "reroll_count",
  "work_step_id", "run_id", "media_seed",
] as const;

type InputSlot = Extract<PromptTemplateSlot, { mode: "input" }>;

function humanize(name: string): string {
  return name.replaceAll("_", " ");
}

function inputSlots(template: PromptTemplateDetail): InputSlot[] {
  return template.current_revision.contract_json.slots.filter(
    (slot): slot is InputSlot => slot.mode === "input",
  );
}

function initialInputs(template: PromptTemplateDetail, count: number): Record<string, string[]> {
  return Object.fromEntries(inputSlots(template).map((slot) => [
    slot.name,
    Array.from({ length: slot.variation_scope === "item" ? count : 1 }, () => ""),
  ]));
}

class PromptBatchAdmissionError extends Error {
  constructor() {
    super("Prompt batch response was invalid.");
    this.name = "PromptBatchAdmissionError";
  }
}

function invalidBatch(): never {
  throw new PromptBatchAdmissionError();
}

function exactRecord(
  value: unknown,
  keys: readonly string[],
): Record<string, unknown> {
  if (
    typeof value !== "object"
    || value === null
    || Array.isArray(value)
    || Object.getPrototypeOf(value) !== Object.prototype
    || Object.getOwnPropertySymbols(value).length !== 0
  ) invalidBatch();
  const record = value as Record<string, unknown>;
  const actual = Object.keys(record);
  if (actual.length !== keys.length || keys.some((key) => !Object.hasOwn(record, key))) {
    invalidBatch();
  }
  return record;
}

function boundedString(value: unknown, maximum: number): string {
  if (
    typeof value !== "string"
    || value.length < 1
    || value.length > maximum
    || value.includes("\0")
  ) invalidBatch();
  for (let index = 0; index < value.length; index += 1) {
    const unit = value.charCodeAt(index);
    if (unit >= 0xd800 && unit <= 0xdbff) {
      const following = value.charCodeAt(index + 1);
      if (!(following >= 0xdc00 && following <= 0xdfff)) invalidBatch();
      index += 1;
    } else if (unit >= 0xdc00 && unit <= 0xdfff) {
      invalidBatch();
    }
  }
  return value;
}

function exactInteger(value: unknown, minimum: number, maximum: number): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    invalidBatch();
  }
  return value as number;
}

function digestString(value: unknown): string {
  const digest = boundedString(value, 64);
  if (!SHA256.test(digest)) invalidBatch();
  return digest;
}

function optionalIdentifier(value: unknown): string | null {
  return value === null ? null : boundedString(value, MAX_IDENTIFIER_CHARACTERS);
}

async function promptDigest(prompt: string): Promise<string> {
  const material = new TextEncoder().encode(`prompt-expansion-rendered-v1\0${prompt}`);
  const digest = await crypto.subtle.digest("SHA-256", material);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

interface BatchAuthority {
  responseKind: "create-or-replay" | "read" | "mutation" | "queue";
  chatId: string;
  templateId: string;
  revisionId: string;
  schemaVersion: number;
  contractSha256: string;
  itemCount: number;
  selectionSeed: number;
  batchId?: string;
  exactPlanVersion?: number;
  queueIdempotencyKey?: string;
}

async function admitPromptBatch(value: unknown, authority: BatchAuthority): Promise<PromptBatch> {
  const raw = exactRecord(value, BATCH_KEYS);
  const id = boundedString(raw.id, MAX_IDENTIFIER_CHARACTERS);
  const chatId = boundedString(raw.chat_id, MAX_IDENTIFIER_CHARACTERS);
  const templateId = boundedString(raw.prompt_template_id, MAX_IDENTIFIER_CHARACTERS);
  const revisionId = boundedString(raw.prompt_template_revision_id, MAX_IDENTIFIER_CHARACTERS);
  const schemaVersion = exactInteger(raw.schema_version, 1, 1);
  const contractSha256 = digestString(raw.contract_sha256);
  const codecVersion = exactInteger(raw.codec_version, 2, 2);
  const requestedCount = exactInteger(raw.requested_count, MIN_ITEMS, MAX_ITEMS);
  const selectionSeed = exactInteger(raw.selection_seed, 0, MAX_SELECTION_SEED);
  const planSha256 = digestString(raw.plan_sha256);
  const state = boundedString(raw.state, 16);
  const planVersion = exactInteger(raw.plan_version, 1, Number.MAX_SAFE_INTEGER);
  const queueIdempotencyKey = raw.queue_idempotency_key === null
    ? null
    : boundedString(raw.queue_idempotency_key, 200);
  const workPlanId = optionalIdentifier(raw.work_plan_id);
  const queuedAt = raw.queued_at === null ? null : boundedString(raw.queued_at, 64);
  const expectedState = authority.responseKind === "queue" ? "queued" : "draft";
  if (typeof raw.replayed !== "boolean") invalidBatch();
  const replayed = raw.replayed;
  if (
    chatId !== authority.chatId
    || templateId !== authority.templateId
    || revisionId !== authority.revisionId
    || schemaVersion !== authority.schemaVersion
    || contractSha256 !== authority.contractSha256
    || codecVersion !== 2
    || requestedCount !== authority.itemCount
    || selectionSeed !== authority.selectionSeed
    || state !== expectedState
    || (authority.responseKind === "create-or-replay" && !replayed && planVersion !== 1)
    || (authority.responseKind === "read" && !replayed)
    || (authority.responseKind === "mutation" && replayed)
    || (
      authority.responseKind === "queue"
      && queueIdempotencyKey !== authority.queueIdempotencyKey
    )
    || (authority.batchId !== undefined && id !== authority.batchId)
    || (authority.exactPlanVersion !== undefined && planVersion !== authority.exactPlanVersion)
  ) invalidBatch();
  if (
    !Array.isArray(raw.items)
    || Object.getPrototypeOf(raw.items) !== Array.prototype
    || raw.items.length !== requestedCount
    || raw.items.length > MAX_ITEMS
  ) invalidBatch();

  const itemIds = new Set<string>();
  const items = await Promise.all(raw.items.map(async (candidate, index) => {
    const entry = exactRecord(candidate, ITEM_KEYS);
    const itemId = boundedString(entry.id, MAX_IDENTIFIER_CHARACTERS);
    const ordinal = exactInteger(entry.ordinal, 1, MAX_ITEMS);
    const renderedPrompt = boundedString(entry.rendered_prompt, MAX_REVIEW_CHARACTERS);
    const renderedSha256 = digestString(entry.rendered_sha256);
    const reviewedPrompt = boundedString(entry.reviewed_prompt, MAX_REVIEW_CHARACTERS);
    const reviewedSha256 = digestString(entry.reviewed_sha256);
    const reviewVersion = exactInteger(entry.review_version, 1, Number.MAX_SAFE_INTEGER);
    const rerollCount = exactInteger(entry.reroll_count, 0, Number.MAX_SAFE_INTEGER);
    const workStepId = optionalIdentifier(entry.work_step_id);
    const runId = optionalIdentifier(entry.run_id);
    const mediaSeed = entry.media_seed === null
      ? null
      : exactInteger(entry.media_seed, 0, MAX_SELECTION_SEED);
    if (
      ordinal !== index + 1
      || typeof entry.selected !== "boolean"
      || itemIds.has(itemId)
    ) invalidBatch();
    itemIds.add(itemId);
    const [computedRendered, computedReviewed] = await Promise.all([
      promptDigest(renderedPrompt),
      promptDigest(reviewedPrompt),
    ]);
    if (renderedSha256 !== computedRendered || reviewedSha256 !== computedReviewed) {
      invalidBatch();
    }
    return {
      id: itemId,
      ordinal,
      rendered_prompt: renderedPrompt,
      rendered_sha256: renderedSha256,
      reviewed_prompt: reviewedPrompt,
      reviewed_sha256: reviewedSha256,
      selected: entry.selected,
      review_version: reviewVersion,
      reroll_count: rerollCount,
      work_step_id: workStepId,
      run_id: runId,
      media_seed: mediaSeed,
    };
  }));
  if (
    state === "draft"
    && (
      queueIdempotencyKey !== null
      || workPlanId !== null
      || queuedAt !== null
      || items.some((item) =>
        item.work_step_id !== null || item.run_id !== null || item.media_seed !== null)
    )
  ) invalidBatch();
  if (state === "queued") {
    if (
      queueIdempotencyKey === null
      || workPlanId === null
      || queuedAt === null
      || Number.isNaN(Date.parse(queuedAt))
      || !items.some((item) => item.selected)
    ) invalidBatch();
    const workStepIds = new Set<string>();
    const runIds = new Set<string>();
    const mediaSeeds = new Set<number>();
    for (const item of items) {
      if (!item.selected) {
        if (item.work_step_id !== null || item.run_id !== null || item.media_seed !== null) {
          invalidBatch();
        }
        continue;
      }
      if (
        item.work_step_id === null
        || item.run_id === null
        || item.media_seed === null
        || workStepIds.has(item.work_step_id)
        || runIds.has(item.run_id)
        || mediaSeeds.has(item.media_seed)
      ) invalidBatch();
      workStepIds.add(item.work_step_id);
      runIds.add(item.run_id);
      mediaSeeds.add(item.media_seed);
    }
  }

  return {
    id,
    chat_id: chatId,
    prompt_template_id: templateId,
    prompt_template_revision_id: revisionId,
    schema_version: schemaVersion,
    contract_sha256: contractSha256,
    codec_version: codecVersion,
    requested_count: requestedCount,
    selection_seed: selectionSeed,
    plan_sha256: planSha256,
    state: state as PromptBatch["state"],
    plan_version: planVersion,
    queue_idempotency_key: queueIdempotencyKey,
    work_plan_id: workPlanId,
    queued_at: queuedAt,
    items,
    replayed,
  };
}

function batchAuthority(batch: PromptBatch, exactPlanVersion: number): BatchAuthority {
  return {
    responseKind: "mutation",
    chatId: batch.chat_id,
    templateId: batch.prompt_template_id,
    revisionId: batch.prompt_template_revision_id,
    schemaVersion: batch.schema_version,
    contractSha256: batch.contract_sha256,
    itemCount: batch.requested_count,
    selectionSeed: batch.selection_seed,
    batchId: batch.id,
    exactPlanVersion,
  };
}

function sameBatchExceptReplay(left: PromptBatch, right: PromptBatch): boolean {
  return left.id === right.id
    && left.chat_id === right.chat_id
    && left.prompt_template_id === right.prompt_template_id
    && left.prompt_template_revision_id === right.prompt_template_revision_id
    && left.schema_version === right.schema_version
    && left.contract_sha256 === right.contract_sha256
    && left.codec_version === right.codec_version
    && left.requested_count === right.requested_count
    && left.selection_seed === right.selection_seed
    && left.plan_sha256 === right.plan_sha256
    && left.state === right.state
    && left.plan_version === right.plan_version
    && left.queue_idempotency_key === right.queue_idempotency_key
    && left.work_plan_id === right.work_plan_id
    && left.queued_at === right.queued_at
    && left.items.length === right.items.length
    && left.items.every((item, index) => {
      const candidate = right.items[index];
      return item.id === candidate.id
        && item.ordinal === candidate.ordinal
        && item.rendered_prompt === candidate.rendered_prompt
        && item.rendered_sha256 === candidate.rendered_sha256
        && item.reviewed_prompt === candidate.reviewed_prompt
        && item.reviewed_sha256 === candidate.reviewed_sha256
        && item.selected === candidate.selected
        && item.review_version === candidate.review_version
        && item.reroll_count === candidate.reroll_count
        && item.work_step_id === candidate.work_step_id
        && item.run_id === candidate.run_id
        && item.media_seed === candidate.media_seed;
    });
}

function patchPreservesBatch(
  previous: PromptBatch,
  updated: PromptBatch,
  ordinal: number,
  reviewedPrompt: string,
  selected: boolean,
): boolean {
  if (
    updated.id !== previous.id
    || updated.chat_id !== previous.chat_id
    || updated.prompt_template_id !== previous.prompt_template_id
    || updated.prompt_template_revision_id !== previous.prompt_template_revision_id
    || updated.schema_version !== previous.schema_version
    || updated.contract_sha256 !== previous.contract_sha256
    || updated.codec_version !== previous.codec_version
    || updated.requested_count !== previous.requested_count
    || updated.selection_seed !== previous.selection_seed
    || updated.state !== previous.state
    || updated.plan_version !== previous.plan_version + 1
    || updated.queue_idempotency_key !== previous.queue_idempotency_key
    || updated.work_plan_id !== previous.work_plan_id
    || updated.queued_at !== previous.queued_at
    || updated.replayed
  ) return false;
  return updated.items.every((item, index) => {
    const before = previous.items[index];
    const unchangedOrigin = item.id === before.id
      && item.ordinal === before.ordinal
      && item.rendered_prompt === before.rendered_prompt
      && item.rendered_sha256 === before.rendered_sha256
      && item.reroll_count === before.reroll_count
      && item.work_step_id === before.work_step_id
      && item.run_id === before.run_id
      && item.media_seed === before.media_seed;
    if (!unchangedOrigin) return false;
    if (item.ordinal === ordinal) {
      const promptChanged = reviewedPrompt !== before.reviewed_prompt;
      const planDigestChanged = updated.plan_sha256 !== previous.plan_sha256;
      return item.review_version === before.review_version + 1
        && item.reviewed_prompt === reviewedPrompt
        && item.selected === selected
        && planDigestChanged === promptChanged;
    }
    return item.review_version === before.review_version
      && item.reviewed_prompt === before.reviewed_prompt
      && item.reviewed_sha256 === before.reviewed_sha256
      && item.selected === before.selected
      && item.work_step_id === before.work_step_id
      && item.run_id === before.run_id
      && item.media_seed === before.media_seed;
  });
}

function queuePreservesBatch(previous: PromptBatch, queued: PromptBatch): boolean {
  return queued.id === previous.id
    && queued.chat_id === previous.chat_id
    && queued.prompt_template_id === previous.prompt_template_id
    && queued.prompt_template_revision_id === previous.prompt_template_revision_id
    && queued.schema_version === previous.schema_version
    && queued.contract_sha256 === previous.contract_sha256
    && queued.codec_version === previous.codec_version
    && queued.requested_count === previous.requested_count
    && queued.selection_seed === previous.selection_seed
    && queued.plan_sha256 === previous.plan_sha256
    && queued.state === "queued"
    && queued.plan_version === previous.plan_version + 1
    && queued.items.length === previous.items.length
    && queued.items.every((item, index) => {
      const before = previous.items[index];
      return item.id === before.id
        && item.ordinal === before.ordinal
        && item.rendered_prompt === before.rendered_prompt
        && item.rendered_sha256 === before.rendered_sha256
        && item.reviewed_prompt === before.reviewed_prompt
        && item.reviewed_sha256 === before.reviewed_sha256
        && item.selected === before.selected
        && item.review_version === before.review_version
        && item.reroll_count === before.reroll_count;
    });
}

function ReviewItemCard({
  batch,
  item,
  authorityCurrent,
  onInsertIntoComposer,
  onDirtyChange,
}: {
  batch: PromptBatch;
  item: PromptBatchItem;
  authorityCurrent: boolean;
  onInsertIntoComposer: (insertion: PromptComposerInsertion) => void;
  onDirtyChange: (itemId: string, dirty: boolean) => void;
}) {
  const client = useQueryClient();
  const [draft, setDraft] = useState({
    sourceVersion: item.review_version,
    reviewedPrompt: item.reviewed_prompt,
    selected: item.selected,
  });
  const [error, setError] = useState<string | null>(null);
  const currentDraft = draft.sourceVersion === item.review_version
    ? draft
    : {
        sourceVersion: item.review_version,
        reviewedPrompt: item.reviewed_prompt,
        selected: item.selected,
      };
  const reviewedPrompt = currentDraft.reviewedPrompt;
  const selected = currentDraft.selected;
  const changedPrompt = reviewedPrompt !== item.reviewed_prompt;
  const changedSelection = selected !== item.selected;
  const dirty = changedPrompt || changedSelection;
  const save = useMutation({
    mutationFn: async () => {
      const payload: PromptBatchItemUpdateInput = {
        expected_review_version: item.review_version,
        expected_plan_version: batch.plan_version,
        reviewed_prompt: reviewedPrompt,
        selected,
      };
      const response = await api.updatePromptBatchItem(batch.id, item.ordinal, payload);
      const admitted = await admitPromptBatch(
        response,
        batchAuthority(batch, batch.plan_version + 1),
      );
      if (!patchPreservesBatch(batch, admitted, item.ordinal, reviewedPrompt, selected)) {
        invalidBatch();
      }
      return admitted;
    },
    onSuccess: (updated) => {
      client.setQueryData<PromptBatch>(["prompt-batch", batch.id], updated);
      setError(null);
    },
    onError: () => {
      setError("This draft could not be saved. Reload the batch before trying again.");
      void client.invalidateQueries({ queryKey: ["prompt-batch", batch.id] });
    },
  });
  useEffect(() => {
    onDirtyChange(item.id, dirty || save.isPending);
    return () => onDirtyChange(item.id, false);
  }, [dirty, item.id, onDirtyChange, save.isPending]);

  return (
    <article className="prompt-review-card" aria-labelledby={`prompt-draft-${item.ordinal}`}>
      <div className="prompt-review-heading">
        <div>
          <small>Draft {item.ordinal}</small>
          <h3 id={`prompt-draft-${item.ordinal}`}>
            {item.selected ? "Selected for later use" : "Not selected"}
          </h3>
        </div>
        <label className="toggle-row">
          <span>Selected</span>
          <input
            type="checkbox"
            checked={selected}
            disabled={!authorityCurrent || save.isPending}
            onChange={(event) => setDraft({
              ...currentDraft,
              selected: event.target.checked,
            })}
          />
        </label>
      </div>
      <details>
        <summary>Original expansion</summary>
        <pre>{item.rendered_prompt}</pre>
      </details>
      <label>
        Reviewed prompt
        <textarea
          aria-label={`Reviewed prompt for draft ${item.ordinal}`}
          rows={5}
          maxLength={MAX_REVIEW_CHARACTERS}
          value={reviewedPrompt}
          disabled={!authorityCurrent || save.isPending}
          onChange={(event) => setDraft({
            ...currentDraft,
            reviewedPrompt: event.target.value,
          })}
        />
      </label>
      <ErrorCallout message={error} />
      <div className="prompt-review-actions">
        <small aria-live="polite">
          {save.isPending ? "Saving review..." : dirty ? "Unsaved review changes" : "Review saved"}
        </small>
        <button
          type="button"
          className="secondary compact-button"
          disabled={!authorityCurrent || !dirty || save.isPending || !reviewedPrompt.trim()}
          onClick={() => {
            setError(null);
            save.mutate();
          }}
        >
          {save.isPending ? <RefreshCw size={14} className="spin" /> : <Check size={14} />}
          Save review
        </button>
        <button
          type="button"
          className="primary compact-button"
          disabled={
            !authorityCurrent
            || dirty
            || save.isPending
            || !item.selected
            || !item.reviewed_prompt.trim()
          }
          onClick={() => onInsertIntoComposer(insertionForPromptBatchItem(batch, item))}
        >
          Insert into composer
        </button>
      </div>
    </article>
  );
}

export function PromptExpansionDialog({
  template,
  activeChatId,
  authorityCurrent,
  onClose,
  onInsertIntoComposer,
}: {
  template: PromptTemplateDetail;
  activeChatId: string;
  authorityCurrent: boolean;
  onClose: () => void;
  onInsertIntoComposer: (insertion: PromptComposerInsertion) => void;
}) {
  const client = useQueryClient();
  const [itemCount, setItemCount] = useState(2);
  const [selectionSeed, setSelectionSeed] = useState(0);
  const [inputs, setInputs] = useState(() => initialInputs(template, 2));
  const [batchId, setBatchId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dirtyItemIds, setDirtyItemIds] = useState<Set<string>>(() => new Set());
  const idempotencyKey = useRef(crypto.randomUUID());
  const queueIdempotencyKey = useRef(crypto.randomUUID());
  const submitting = useRef(false);
  const reviewHeading = useRef<HTMLHeadingElement>(null);
  const authoredSlots = useMemo(() => inputSlots(template), [template]);
  const modelSlots = template.current_revision.contract_json.slots.filter(
    (slot) => slot.mode === "model",
  );
  const batchQuery = useQuery({
    queryKey: ["prompt-batch", batchId],
    queryFn: async () => {
      const authorityBatch = client.getQueryData<PromptBatch>(["prompt-batch", batchId]);
      const response = await api.promptBatch(batchId!);
      const admitted = await admitPromptBatch(response, {
        responseKind: authorityBatch?.state === "queued" ? "queue" : "read",
        chatId: activeChatId,
        templateId: template.id,
        revisionId: template.current_revision.id,
        schemaVersion: template.current_revision.schema_version,
        contractSha256: template.current_revision.contract_sha256,
        itemCount,
        selectionSeed,
        batchId: batchId!,
        exactPlanVersion: authorityBatch?.state === "queued"
          ? authorityBatch.plan_version
          : undefined,
        queueIdempotencyKey: authorityBatch?.queue_idempotency_key ?? undefined,
      });
      const latest = client.getQueryData<PromptBatch>(["prompt-batch", batchId]);
      if (!latest) return admitted;
      if (admitted.plan_version < latest.plan_version) {
        return { ...latest, replayed: true };
      }
      if (
        admitted.plan_version === latest.plan_version
        && !sameBatchExceptReplay(admitted, latest)
      ) invalidBatch();
      return admitted;
    },
    enabled: Boolean(batchId),
  });
  const trustedBatch = !batchQuery.error ? batchQuery.data ?? null : null;
  const trustedBatchId = trustedBatch?.id;
  useEffect(() => {
    if (trustedBatchId) reviewHeading.current?.focus();
  }, [trustedBatchId]);

  const invalidateAuthority = (candidateBatchId?: string) => {
    void client.invalidateQueries({ queryKey: ["prompt-template", template.id] });
    void client.invalidateQueries({ queryKey: ["prompt-template-revisions", template.id] });
    if (candidateBatchId) {
      void client.invalidateQueries({ queryKey: ["prompt-batch", candidateBatchId] });
    }
  };

  const create = useMutation({
    mutationFn: async (payload: PromptBatchCreateInput) => {
      const response = await api.createPromptBatch(activeChatId, payload);
      return admitPromptBatch(response, {
        responseKind: "create-or-replay",
        chatId: activeChatId,
        templateId: template.id,
        revisionId: template.current_revision.id,
        schemaVersion: template.current_revision.schema_version,
        contractSha256: template.current_revision.contract_sha256,
        itemCount,
        selectionSeed,
      });
    },
    onSuccess: (created) => {
      client.setQueryData(["prompt-batch", created.id], created);
      setBatchId(created.id);
      setError(null);
    },
    onError: (failure) => {
      setError(failure instanceof PromptBatchAdmissionError
        ? "The generated batch did not match the selected template revision. Refresh before trying again."
        : "The Prompt Library could not create those drafts. Refresh the template and try again.");
      invalidateAuthority();
    },
    onSettled: () => {
      submitting.current = false;
    },
  });
  const queue = useMutation({
    mutationFn: async (batch: PromptBatch) => {
      const payload: PromptBatchQueueInput = {
        idempotency_key: queueIdempotencyKey.current,
        expected_plan_version: batch.plan_version,
        expected_plan_sha256: batch.plan_sha256,
      };
      const response = await api.queuePromptBatch(batch.id, payload);
      const admitted = await admitPromptBatch(response, {
        responseKind: "queue",
        chatId: batch.chat_id,
        templateId: batch.prompt_template_id,
        revisionId: batch.prompt_template_revision_id,
        schemaVersion: batch.schema_version,
        contractSha256: batch.contract_sha256,
        itemCount: batch.requested_count,
        selectionSeed: batch.selection_seed,
        batchId: batch.id,
        exactPlanVersion: batch.plan_version + 1,
        queueIdempotencyKey: queueIdempotencyKey.current,
      });
      if (!queuePreservesBatch(batch, admitted)) invalidBatch();
      return admitted;
    },
    onSuccess: (queued) => {
      client.setQueryData<PromptBatch>(["prompt-batch", queued.id], queued);
      void client.invalidateQueries({ queryKey: ["chat", activeChatId] });
      void client.invalidateQueries({ queryKey: ["work-plans", activeChatId] });
      setDirtyItemIds(new Set());
      setError(null);
    },
    onError: () => {
      setError(
        "The selected drafts could not be queued atomically. Reload the batch and try again.",
      );
      if (batchId) {
        void client.invalidateQueries({ queryKey: ["prompt-batch", batchId] });
      }
    },
  });

  const updateCount = (next: number) => {
    if (!Number.isInteger(next) || next < MIN_ITEMS || next > MAX_ITEMS) return;
    setItemCount(next);
    setInputs((current) => Object.fromEntries(authoredSlots.map((slot) => {
      const length = slot.variation_scope === "item" ? next : 1;
      const previous = current[slot.name] ?? [];
      return [slot.name, Array.from({ length }, (_, index) => previous[index] ?? "")];
    })));
    idempotencyKey.current = crypto.randomUUID();
    setError(null);
  };

  const updateInput = (slot: InputSlot, index: number, value: string) => {
    setInputs((current) => ({
      ...current,
      [slot.name]: current[slot.name].map((entry, currentIndex) =>
        currentIndex === index ? value : entry),
    }));
    idempotencyKey.current = crypto.randomUUID();
    setError(null);
  };

  const submit = () => {
    if (!authorityCurrent) {
      setError("This template or chat changed while the dialog was open. Close and reopen it.");
      return;
    }
    if (
      !Number.isInteger(itemCount)
      || itemCount < MIN_ITEMS
      || itemCount > MAX_ITEMS
      || !Number.isInteger(selectionSeed)
      || selectionSeed < 0
      || selectionSeed > MAX_SELECTION_SEED
    ) {
      setError("Use a draft count from 1 to 16 and a selection seed from 0 to 2147483647.");
      return;
    }
    if (Object.values(inputs).some((values) =>
      values.some((value) => !value.trim() || value.length > MAX_INPUT_CHARACTERS))) {
      setError("Every authored input needs between 1 and 2,000 characters.");
      return;
    }
    if (submitting.current || create.isPending) return;
    submitting.current = true;
    const serializedInputs: Record<string, string | string[]> = {};
    for (const slot of authoredSlots) {
      serializedInputs[slot.name] = slot.variation_scope === "item"
        ? inputs[slot.name].slice(0, itemCount)
        : inputs[slot.name][0];
    }
    create.mutate({
      idempotency_key: idempotencyKey.current,
      template_revision_id: template.current_revision.id,
      contract_sha256: template.current_revision.contract_sha256,
      item_count: itemCount,
      selection_seed: selectionSeed,
      inputs: serializedInputs,
    });
  };

  const visibleError = !authorityCurrent
    ? "This template or chat changed while the dialog was open. Close and reopen it."
    : error
      ?? (queue.error
        ? "The selected drafts could not be queued atomically. Reload the batch and try again."
        : batchQuery.error
          ? "The saved draft batch could not be loaded. Refresh and try again."
          : null);
  const updateDirtyItem = useCallback((itemId: string, dirty: boolean) => {
    setDirtyItemIds((current) => {
      if (current.has(itemId) === dirty) return current;
      const next = new Set(current);
      if (dirty) next.add(itemId);
      else next.delete(itemId);
      return next;
    });
  }, []);

  return (
    <AccessibleDialog
      title={`Test ${template.name}`}
      eyebrow="No-media prompt expansion"
      closeLabel="Close prompt expansion"
      className="prompt-expansion-dialog"
      onClose={onClose}
    >
      <ErrorCallout message={visibleError} />
      {!trustedBatch && (
        <>
          <p className="prompt-expansion-note">
            Generate editable prompt drafts in the active chat. This test does not queue images or video.
          </p>
          <div className="prompt-expansion-controls">
            <label>
              Draft count
              <input
                aria-label="Draft count"
                type="number"
                min={MIN_ITEMS}
                max={MAX_ITEMS}
                step={1}
                value={itemCount}
                disabled={!authorityCurrent || create.isPending}
                onChange={(event) => updateCount(event.target.valueAsNumber)}
              />
            </label>
            <label>
              Selection seed
              <input
                aria-label="Selection seed"
                type="number"
                min={0}
                max={MAX_SELECTION_SEED}
                step={1}
                value={selectionSeed}
                disabled={!authorityCurrent || create.isPending}
                onChange={(event) => {
                  setSelectionSeed(event.target.valueAsNumber);
                  idempotencyKey.current = crypto.randomUUID();
                  setError(null);
                }}
              />
            </label>
          </div>
          {authoredSlots.length > 0 && (
            <section className="prompt-expansion-inputs" aria-labelledby="prompt-expansion-input-heading">
              <h3 id="prompt-expansion-input-heading">Authored inputs</h3>
              {authoredSlots.flatMap((slot) => inputs[slot.name].map((value, index) => (
                <label key={`${slot.name}-${index}`}>
                  {humanize(slot.name)}
                  {slot.variation_scope === "item" ? ` - draft ${index + 1}` : " - shared across batch"}
                  <textarea
                    aria-label={`${humanize(slot.name)} ${slot.variation_scope === "item" ? `for draft ${index + 1}` : "shared across batch"}`}
                    rows={2}
                    maxLength={MAX_INPUT_CHARACTERS}
                    value={value}
                    disabled={!authorityCurrent || create.isPending}
                    onChange={(event) => updateInput(slot, index, event.target.value)}
                  />
                </label>
              )))}
            </section>
          )}
          {modelSlots.length > 0 && (
            <section className="prompt-model-guidance" aria-labelledby="prompt-model-guidance-heading">
              <h3 id="prompt-model-guidance-heading">Model-guided slots</h3>
              <p>The backend resolves these slots; guidance is read-only here.</p>
              <dl>
                {modelSlots.map((slot) => (
                  <div key={slot.name}>
                    <dt>{humanize(slot.name)}</dt>
                    <dd>{slot.guidance}</dd>
                  </div>
                ))}
              </dl>
            </section>
          )}
          <footer>
            <button type="button" className="secondary" onClick={onClose}>Cancel</button>
            <button
              type="button"
              className="primary"
              disabled={!authorityCurrent || create.isPending}
              onClick={submit}
            >
              {create.isPending ? "Generating drafts..." : "Generate drafts"}
            </button>
          </footer>
        </>
      )}
      {trustedBatch && (
        <section className="prompt-review-list" aria-labelledby="prompt-review-heading">
          <div className="section-heading compact-heading">
            <div>
              <h3 ref={reviewHeading} id="prompt-review-heading" tabIndex={-1}>Review drafts</h3>
              <p>
                Plan {trustedBatch.plan_sha256.slice(0, 12)}
                {trustedBatch.replayed ? " - replayed safely" : ""}
              </p>
            </div>
          </div>
          <p className="prompt-review-status" role="status">
            {trustedBatch.items.length} drafts ready to review.
          </p>
          {trustedBatch.items.map((item) => (
            <ReviewItemCard
              key={item.id}
              batch={trustedBatch}
              item={item}
              authorityCurrent={
                authorityCurrent && trustedBatch.state === "draft" && !queue.isPending
              }
              onInsertIntoComposer={onInsertIntoComposer}
              onDirtyChange={updateDirtyItem}
            />
          ))}
          <footer>
            <button type="button" className="secondary" onClick={onClose}>Done</button>
            {trustedBatch.state === "draft" && (
              <button
                type="button"
                className="primary"
                disabled={
                  !authorityCurrent
                  || queue.isPending
                  || dirtyItemIds.size > 0
                  || !trustedBatch.items.some((item) => item.selected)
                }
                onClick={() => queue.mutate(trustedBatch)}
              >
                {queue.isPending ? "Queueing selected drafts..." : "Queue selected drafts"}
              </button>
            )}
          </footer>
          {trustedBatch.state === "queued" && (
            <p className="prompt-review-status" role="status">
              {trustedBatch.items.filter((item) => item.selected).length}
              {" "}selected drafts were queued as one work plan.
            </p>
          )}
        </section>
      )}
    </AccessibleDialog>
  );
}
